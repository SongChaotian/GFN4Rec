import numpy as np
import utils
import torch
import random
from copy import deepcopy
from argparse import Namespace
from torch.utils.data import DataLoader
from torch.distributions import Categorical

from reader import *
from model.simulator import *


class KRUserEnvironment_FiniteImmediate():
    '''
    KuaiRand模拟环境（GPU版本）
    
    核心组件：
    1. 多行为用户响应模型：(用户历史, 用户画像) -> 用户状态 -> (状态, 物品) -> 反馈
    2. 用户离开模型：temper值降至<1时离开，每步递减且不满意时额外下降
    '''
    
    @staticmethod
    def parse_model_args(parser):
        '''环境参数配置'''
        parser.add_argument('--uirm_log_path', type=str, required=True, 
                            help='用户响应模型的日志路径')
        parser.add_argument('--initial_temper', type=int, required=10, 
                            help='用户初始耐心值')
        parser.add_argument('--slate_size', type=int, required=6, 
                            help='每次推荐的物品数量')
        parser.add_argument('--max_step_per_episode', type=int, default=30, 
                            help='每个episode的最大步数')
        parser.add_argument('--episode_batch_size', type=int, default=32, 
                            help='并行运行的用户数')
        parser.add_argument('--item_correlation', type=float, default=0, 
                            help='物品相关性惩罚系数')
        parser.add_argument('--new_reader_class', type=str, default='', 
                            help='离线训练时使用的数据读取器')
        parser.add_argument('--env_val_holdout', type=int, default=0, 
                            help='验证集保留数量')
        parser.add_argument('--env_test_holdout', type=int, default=0, 
                            help='测试集保留数量')
        return parser
    
    def __init__(self, args):
        super().__init__()
        self.device = args.device
        
        # 环境参数
        self.initial_temper = args.initial_temper
        self.slate_size = args.slate_size
        self.max_step_per_episode = args.max_step_per_episode
        self.episode_batch_size = args.episode_batch_size
        self.rho = args.item_correlation  # 物品相关性系数
        
        # 加载用户响应模型
        print("Load immediate user response model")
        uirm_stats, uirm_model, uirm_args = self.get_user_model(args.uirm_log_path, args.device)
        self.immediate_response_stats = uirm_stats
        self.immediate_response_model = uirm_model
        self.max_hist_len = uirm_stats['max_seq_len']
        self.response_types = uirm_stats['feedback_type']
        self.response_dim = len(self.response_types)
        self.response_weights = [0 if f == 'is_hate' else 1 for f in self.response_types]
        
        # 加载数据读取器
        print("Load user sequence reader")
        reader, reader_args = self.get_reader(args)
        self.reader = reader
        print(self.reader.get_statistics())
        
        # 构建候选物品池
        print("Setup candidate item pool")
        # 物品ID编码 (n_item,)
        self.candidate_iids = torch.tensor(
            [reader.item_id_vocab[iid] for iid in reader.items]
        ).to(self.device)
        
        # 物品特征 {'feature_name': (n_item, feature_dim)}
        candidate_meta = [reader.get_item_meta_data(iid) for iid in reader.items]
        self.candidate_item_meta = {}
        self.n_candidate = len(candidate_meta)
        for k in candidate_meta[0]:
            self.candidate_item_meta[k[3:]] = torch.FloatTensor(
                np.concatenate([meta[k] for meta in candidate_meta])
            ).view(self.n_candidate, -1).to(self.device)
        
        # 物品编码 (n_item, item_enc_dim)
        item_enc, _ = self.immediate_response_model.get_item_encoding(
            self.candidate_iids, 
            {k: v for k, v in self.candidate_item_meta.items()}, 
            1
        )
        self.candidate_item_encoding = torch.clamp(item_enc, -1, 1).view(
            -1, self.immediate_response_model.enc_dim
        )
        
        # 定义状态和动作空间
        self.gt_state_dim = self.immediate_response_model.state_dim
        self.action_dim = self.slate_size
        self.observation_space = self.reader.get_statistics()
        self.action_space = self.n_candidate
        
        self.immediate_response_model.to(args.device)
        self.immediate_response_model.device = args.device
        self.env_response_history = {}
        
    def get_candidate_info(self, observation):
        '''
        获取候选物品信息
        @output: {'item_id': (1, n_item), 'item_{feature}': (1, n_item, feature_dim)}
        '''
        candidate_info = {'item_id': self.candidate_iids.view(1, -1)}
        candidate_info.update({
            f'item_{k}': v.view(1, len(self.candidate_iids), -1) 
            for k, v in self.candidate_item_meta.items()
        })
        return candidate_info
        
    def get_user_model(self, log_path, device, from_load=True):
        '''从日志路径加载用户响应模型'''
        # 读取训练配置
        infile = open(log_path, 'r')
        class_args = eval(infile.readline())
        model_args = eval(infile.readline())
        infile.close()
        
        # 加载模型
        checkpoint = torch.load(
            model_args.model_path + ".checkpoint", 
            map_location=device, 
            weights_only=False
        )
        reader_stats = checkpoint["reader_stats"]
        modelClass = eval('{0}.{0}'.format(class_args.model))
        model = modelClass(model_args, reader_stats, device)
        
        if from_load:
            model.load_from_checkpoint(model_args.model_path, with_optimizer=False)
        model = model.to(device)
        
        return reader_stats, model, model_args
    
    def get_reader(self, args):
        '''加载数据读取器'''
        # 读取训练配置
        infile = open(args.uirm_log_path, 'r')
        class_args = eval(infile.readline())
        training_args = eval(infile.readline())
        training_args.val_holdout_per_user = args.env_val_holdout
        training_args.test_holdout_per_user = args.env_test_holdout
        training_args.device = self.device
        training_args.slate_size = args.slate_size
        infile.close()
        
        # 实例化读取器
        if len(args.new_reader_class) > 0:
            readerClass = eval('{0}.{0}'.format(args.new_reader_class))
        else:
            readerClass = eval('{0}.{0}'.format(class_args.reader))
        reader = readerClass(training_args)
        
        return reader, training_args
        
    def reset(self):
        '''
        重置环境，采样新用户
        @output: observation字典，包含用户画像和历史
        '''
        BS = self.episode_batch_size
        self.iter = iter(DataLoader(
            self.reader, 
            batch_size=BS, 
            shuffle=True, 
            pin_memory=True, 
            num_workers=8
        ))
        
        initial_sample = next(self.iter)
        self.current_observation = self.get_observation_from_batch(initial_sample)
        
        # 初始化记录
        self.rec_history = {'coverage': [], 'intra_diversity': []}
        self.temper = torch.ones(self.episode_batch_size).to(self.device) * self.initial_temper
        self.user_step_count = 0
        
        return deepcopy(self.current_observation)
    
    def get_observation_from_batch(self, sample_batch):
        '''
        从数据批次构造观察
        @input: 包含用户ID、特征、历史的字典
        @output: {'user_profile': {...}, 'user_history': {...}}
        '''
        sample_batch = utils.wrap_batch(sample_batch, device=self.device)
        
        # 提取用户画像
        profile = {'user_id': sample_batch['user_id']}
        for k, v in sample_batch.items():
            if 'uf_' in k:
                profile[k] = v
        
        # 提取用户历史
        history = {'history': sample_batch['history']}
        for k, v in sample_batch.items():
            if 'history_' in k:
                history[k] = v
        
        return {'user_profile': profile, 'user_history': history}
    
    def step(self, step_dict):
        '''
        执行一步交互
        @input: {'action': (B, slate_size)} 推荐的物品索引
        @output: (新观察, 用户反馈, 更新后的观察)
        '''
        action = step_dict['action']  # (B, slate_size)
        
        # 获取用户响应
        with torch.no_grad():
            response_out = self.get_response(step_dict)
            response = response_out['immediate_response']  # (B, slate_size, n_feedback)
            
            # 判断用户是否离开
            done_mask = self.get_leave_signal(response)  # (B,)
            
            # 更新观察
            update_info = self.update_observation(action, response, done_mask)
            self.user_step_count += 1
            
            # 所有用户同时结束时，采样新用户
            if done_mask.sum() == len(done_mask):
                try:
                    sample_info = next(self.iter)
                    if sample_info['user_profile'].shape[0] != len(done_mask):
                        raise StopIteration
                except:
                    self.iter = iter(DataLoader(
                        self.reader, 
                        batch_size=done_mask.shape[0], 
                        shuffle=True, 
                        pin_memory=True, 
                        num_workers=8
                    ))
                    sample_info = next(self.iter)
                
                new_observation = self.get_observation_from_batch(sample_info)
                self.current_observation = new_observation
                self.temper = torch.ones(self.episode_batch_size).to(self.device) * self.initial_temper
                self.user_step_count = 0
                
            elif done_mask.sum() > 0:
                print(done_mask)
                print("User leave not synchronized")
                raise NotImplementedError
        
        # 记录环境报告
        report = self.get_env_report()
        for key, value in report.items():
            if key not in self.env_response_history:
                self.env_response_history[key] = []
            self.env_response_history[key].append(value)
        
        user_feedback = {
            'immediate_response': response, 
            'done': done_mask
        }
        
        return deepcopy(self.current_observation), user_feedback, update_info['updated_observation']
    
    def get_response(self, step_dict):
        '''
        模拟用户响应
        @input: {'action': (B, slate_size)}
        @output: {'immediate_response': (B, slate_size, n_feedback), 'coverage': scalar, 'ILD': scalar}
        '''
        action = step_dict['action']  # (B, slate_size)
        coverage = len(torch.unique(action))
        B = action.shape[0]
        
        # 获取用户状态 (B, 1, gt_state_dim)
        profile_dict = {k: v for k, v in self.current_observation['user_profile'].items()}
        user_state = self.get_ground_truth_user_state(
            profile_dict,
            {k: v for k, v in self.current_observation['user_history'].items()}
        )
        user_state = user_state.view(self.episode_batch_size, 1, self.gt_state_dim)
        
        # 获取推荐物品的编码 (B, slate_size, 1, item_enc_dim)
        selected_item_enc = self.candidate_item_encoding[action].view(
            B, self.slate_size, 1, self.immediate_response_model.enc_dim
        )
        
        # 计算点击分数 (B, slate_size, n_feedback)
        behavior_scores, _ = self.immediate_response_model.get_pointwise_scores(
            user_state, selected_item_enc, self.episode_batch_size
        )
        
        # 计算物品相关性惩罚 (B, slate_size)
        corr_factor = self.get_intra_slate_similarity(
            selected_item_enc.view(B, self.slate_size, -1)
        )
        
        # 采样用户响应
        point_scores = torch.sigmoid(behavior_scores) - corr_factor.view(B, self.slate_size, 1) * self.rho
        point_scores[point_scores < 0] = 0
        response = torch.bernoulli(point_scores)  # (B, slate_size, n_feedback)
        
        return {
            'immediate_response': response, 
            'coverage': coverage,
            'ILD': 1 - torch.mean(corr_factor).item()
        }
    
    def get_ground_truth_user_state(self, profile, history):
        '''获取真实用户状态编码'''
        batch_data = {}
        batch_data.update(profile)
        batch_data.update(history)
        gt_state_dict = self.immediate_response_model.encode_state(
            batch_data, self.episode_batch_size
        )
        gt_user_state = gt_state_dict['state'].view(
            self.episode_batch_size, 1, self.gt_state_dim
        )
        return gt_user_state
    
    def get_intra_slate_similarity(self, action_item_encoding):
        '''
        计算推荐列表内物品相似度
        @input: (B, slate_size, enc_dim)
        @output: (B, slate_size) 每个物品与列表平均的相似度
        '''
        B, L, d = action_item_encoding.shape
        # 两两相似度 (B, L, L)
        pair_similarity = torch.mean(
            action_item_encoding.view(B, L, 1, d) * action_item_encoding.view(B, 1, L, d), 
            dim=-1
        )
        # 与列表平均的相似度 (B, L)
        point_similarity = torch.mean(pair_similarity, dim=-1)
        return point_similarity
    
    def update_observation(self, action, slate_response, done_mask):
        '''
        更新用户观察（添加新交互到历史）
        @input:
        - action: (B, slate_size) 推荐的物品索引
        - slate_response: (B, slate_size, n_feedback) 用户反馈
        - done_mask: (B,) 是否结束
        '''
        rec_list = self.candidate_iids[action]  # (B, slate_size)
        
        old_history = self.current_observation['user_history']
        max_H = self.max_hist_len
        
        # 更新历史长度
        L = old_history['history_length'] + self.slate_size
        L[L > max_H] = max_H
        
        # 更新历史物品ID
        new_history = {
            'history': torch.cat((old_history['history'], rec_list), dim=1)[:, -max_H:], 
            'history_length': L
        }
        
        # 更新物品特征历史
        for f in self.reader.selected_item_features:
            candidate_meta_features = self.candidate_item_meta[f]  # (n_item, feature_dim)
            meta_features = candidate_meta_features[action]  # (B, slate_size, feature_dim)
            k = f'history_if_{f}'
            previous_meta = old_history[k].view(
                self.episode_batch_size, self.observation_space['max_seq_len'], -1
            )
            new_history[k] = torch.cat((previous_meta, meta_features), dim=1)[:, -max_H:, :].view(
                self.episode_batch_size, -1
            )
        
        # 更新反馈历史
        for i, response in enumerate(self.immediate_response_model.feedback_types):
            k = f'history_{response}'
            new_history[k] = torch.cat(
                (old_history[k], slate_response[:, :, i]), dim=1
            )[:, -max_H:]
        
        self.current_observation['user_history'] = new_history
        
        return {
            'slate': rec_list, 
            'updated_observation': deepcopy(self.current_observation)
        }
    
    def get_leave_signal(self, response):
        '''
        判断用户是否离开
        @input: response (B, slate_size, n_feedback)
        @output: done_mask (B,) 布尔张量
        '''
        temper_down = 1
        self.temper -= temper_down
        done_mask = self.temper < 1
        return done_mask
    
    def create_observation_buffer(self, buffer_size):
        '''创建空的观察缓冲区'''
        observation = {
            'user_profile': {
                'user_id': torch.zeros(buffer_size).to(torch.long).to(self.device)
            }, 
            'user_history': {
                'history': torch.zeros(buffer_size, self.max_hist_len).to(torch.long).to(self.device), 
                'history_length': torch.zeros(buffer_size).to(torch.long).to(self.device)
            }
        }
        
        # 添加用户特征
        for f, f_dim in self.observation_space['user_feature_dims'].items():
            observation['user_profile'][f'uf_{f}'] = torch.zeros(
                buffer_size, f_dim
            ).to(torch.float).to(self.device)
        
        # 添加物品特征历史
        for f, f_dim in self.observation_space['item_feature_dims'].items():
            observation['user_history'][f'history_if_{f}'] = torch.zeros(
                buffer_size, f_dim * self.max_hist_len
            ).to(torch.float).to(self.device)
        
        # 添加反馈历史
        for f in self.observation_space['feedback_type']:
            observation['user_history'][f'history_{f}'] = torch.zeros(
                buffer_size, self.max_hist_len
            ).to(torch.float).to(self.device)
        
        return observation
    
    def stop(self):
        '''停止环境'''
        self.iter = None
    
    def get_new_iterator(self, B):
        '''创建新的数据迭代器'''
        return iter(DataLoader(
            self.reader, 
            batch_size=B, 
            shuffle=True, 
            pin_memory=True, 
            num_workers=8
        ))
    
    def get_env_report(self, window=50):
        '''
        获取环境统计报告
        @input: window 统计窗口大小
        @output: 各类反馈的平均值字典
        '''
        report = {}
        for key in self.response_types:
            history_key = f'history_{key}'
            if history_key in self.current_observation['user_history']:
                report[key] = torch.mean(
                    self.current_observation['user_history'][history_key][:, -window:]
                ).item()
            else:
                report[key] = 0
        
        return report
