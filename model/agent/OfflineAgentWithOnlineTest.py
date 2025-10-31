"""
OfflineAgentWithOnlineTest - 离线训练 + 在线评估

功能：使用离线数据集训练，同时在模拟器中评估性能
与在线训练的核心区别：
    1. 数据来源：DataLoader加载离线数据 vs Buffer存储在线交互
    2. 训练方式：监督学习 vs 强化学习
    3. 探索策略：不需要探索 vs ε-greedy探索
    4. 评估方式：离线指标(NDCG/MRR) + 在线评估 vs 仅在线评估
"""

import time
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import utils
from model.agent.reward_func import *
from model.agent.BaseOnlineAgent import BaseOnlineAgent


class OfflineAgentWithOnlineTest(BaseOnlineAgent):
    """
    离线训练Agent（继承自BaseOnlineAgent）
    
    训练流程：
        1. action_before_train(): 初始化DataLoader
        2. 主循环: for i in range(n_iter):
            a) step_train(): 从DataLoader采样批次，监督学习训练
            b) online_one_step_eval(): 在模拟器中评估（不用于训练）
        3. action_after_train(): 在测试集上评估离线指标
    """
    
    @staticmethod
    def parse_model_args(parser):
        """解析命令行参数（继承父类参数）"""
        parser = BaseOnlineAgent.parse_model_args(parser)
        return parser
    
    def __init__(self, *input_args):
        """
        初始化Agent
        
        额外配置：
            - NDCG/MRR折扣因子：用于计算排序指标
            - 反馈类型和权重：多种用户反馈的加权
        """
        args, actor, env, buffer = input_args
        super().__init__(args, actor, env, buffer)
        
        # 离线评估配置
        # 用户反馈类型（如点击、点赞、收藏等）
        self.response_types = self.env.immediate_response_model.feedback_types
        self.W = self.immediate_response_weight.detach().cpu().numpy()
        self.W_sum = sum(self.W)
        
        # NDCG计算的折扣因子：exp(-position)
        self.NDCG_discount = (-torch.arange(self.batch_size)).exp().to(self.device) + 1e-6
        # MRR计算的位置折扣：1/position
        self.rank_discount = (torch.arange(self.batch_size) + 1).pow(-1.0).to(self.device)
    
    def train(self):
        """
        【主训练循环】离线训练流程
        
        与在线训练的区别：
            1. 不调用run_episode_step()进行在线交互
            2. step_train()直接从DataLoader读取离线数据
            3. 训练结束后在测试集上评估离线指标
        """
        # 如果是继续训练，加载之前的模型
        if len(self.n_iter) > 2:
            self.load()
        
        t = time.time()
        print("Run procedures before training")
        self.action_before_train()  # 初始化DataLoader（不返回observation）
        t = time.time()
        start_time = t
        
        # 主训练循环
        print("Training:")
        step_offset = sum(self.n_iter[:-1])
        for i in tqdm(range(step_offset, step_offset + self.n_iter[-1])):
            self.epsilon = self.exploration_scheduler.value(i)
            
            # 【核心差异】直接离线训练，不与环境交互
            self.step_train()  # 从DataLoader采样 + 监督学习训练
            
            # 日志记录
            if i % self.check_episode == 0 and i >= self.check_episode:
                t_prime = time.time()
                episode_report, train_report = self.get_report()
                
                log_str = (
                    f"\n{'='*80}\n"
                    f"Step {i:6d} | Interval: {t_prime-t:6.1f}s | Total: {t_prime-start_time:8.1f}s\n"
                    f"{'-'*80}\n"
                    f"Episode: {episode_report}\n"
                    f"Training: {train_report}\n"
                    f"{'='*80}\n"
                )

                print(log_str)
                with open(self.save_path + ".report", 'a') as outfile:
                    outfile.write(log_str)

                t = t_prime
            
            # 保存模型
            if i % self.save_episode == 0:
                self.save()
        
        # 训练结束：离线评估
        self.action_after_train()
    
    def action_before_train(self):
        """
        训练前准备
        
        与在线训练的区别：
            1. 不初始化Buffer（不需要）
            2. 不进行随机探索填充Buffer
            3. 【关键】初始化DataLoader读取离线数据集
        
        Returns:
            None（不返回observation，因为不需要与环境交互）
        """
        # 训练记录初始化
        self.training_history = {}
        self.eval_history = {
            'avg_reward': [],           # 平均奖励
            'max_reward': [],           # 最大奖励
            'reward_variance': [],      # 奖励方差
            'coverage': [],             # 覆盖率
            'intra_slate_diversity': [], # 列表内多样性
            'NDCG': [],                 # 归一化折扣累积增益
            'MRR': []                   # 平均倒数排名
        }
        # 为每种用户反馈类型添加记录
        self.eval_history.update({f'{resp}_rate': [] for resp in self.env.response_types})
        
        # 为每个位置的NDCG和MRR添加记录
        K = self.env.action_dim  # slate size
        self.eval_history.update({f'NDCG_{t}': [] for t in range(K)})
        self.eval_history.update({f'MRR_{t}': [] for t in range(K)})
        
        self.initialize_training_history()
        
        # 【关键】初始化离线数据迭代器
        # 使用DataLoader加载预先收集的离线数据集
        self.offline_iter = iter(DataLoader(
            self.env.reader,          # MLSlateReader: 离线数据读取器
            batch_size=self.batch_size,
            shuffle=True,             # 打乱数据
            pin_memory=True,          # 固定内存加速GPU传输
            num_workers=8             # 多进程加载
        ))
        
        # 设置数据读取器为训练阶段
        self.env.reader.set_phase('train')
        
        return None  # 注意：不返回observation（离线训练不需要）
    
    def action_after_train(self):
        """
        训练结束后处理
        
        额外操作：
            - 在测试集上评估离线指标（NDCG、MRR）
        """
        self.env.stop()
        # 离线测试
        self.test('test')
    
    def step_train(self):
        """
        【核心】训练步：离线数据监督学习
        
        与在线训练的核心区别：
            1. 数据来源：从DataLoader读取离线数据 vs 从Buffer采样在线交互数据
            2. 训练方式：使用离线数据中的动作和反馈作为监督标签
            3. 额外操作：每步训练后进行在线评估（仅评估，不用于训练）
        
        流程：
            1. 从DataLoader获取批次数据
            2. 提取监督标签（动作、反馈）
            3. 前向传播 + 计算损失
            4. 反向传播更新参数
            5. 在线评估（不更新模型）
        
        离线数据格式：
            batch_sample = {
                'user_id': (B,)
                'item_id': (B, slate_size)              # 监督标签：推荐列表
                'uf_{feature}': (B, F_dim)              # 用户特征
                'if_{feature}': (B, slate_size, F_dim)  # 物品特征
                '{response}': (B, slate_size)           # 监督标签：用户反馈
                'history': (B, max_H)                   # 历史交互
                'history_length': (B,)
                'history_if_{feature}': (B, max_H * F_dim)
                'history_{response}': (B, max_H)
            }
        """
        # 【核心1】从离线数据集获取批次
        try:
            batch_sample = next(self.offline_iter)
        except StopIteration:
            # 数据集遍历完后，重新创建迭代器
            self.offline_iter = iter(DataLoader(
                self.env.reader,
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=8
            ))
            batch_sample = next(self.offline_iter)
        
        # 构造观察
        B = batch_sample['user_id'].shape[0]
        self.env.episode_batch_size = B
        sample_observation = self.env.get_observation_from_batch(batch_sample)
        
        # 【核心2】提取监督标签
        # 目标动作（监督标签）：离线数据中的推荐列表
        # (B, slate_size)
        target_action = batch_sample['item_id'] - 1
        
        # 目标反馈（监督标签）：离线数据中的用户反馈
        # (B, slate_size, response_dim)
        target_response = torch.cat([
            batch_sample[resp].view(B, self.env.action_dim, 1)
            for resp in self.response_types
        ], dim=2)
        
        # 构造用户反馈字典
        user_feedback = {
            'immediate_response': target_response,
            'immediate_response_weight': self.immediate_response_weight
        }
        user_feedback['reward'] = self.reward_func(user_feedback).detach()
        
        # 【核心3】前向传播（监督学习）
        sample_observation['batch_size'] = B
        candidate_info = self.env.get_candidate_info(None)  # 全物品集
        
        input_dict = {
            'observation': sample_observation,
            'candidates': candidate_info,
            'action_dim': self.env.action_dim,
            'action': target_action,          # 使用离线数据的动作（监督标签）
            'response': target_response,      # 使用离线数据的反馈（监督标签）
            'reward': user_feedback['reward'],
            'epsilon': 0,                     # 不探索（监督学习）
            'do_explore': False,              # 不探索
            'is_train': True                  # 训练模式
        }
        policy_output = self.actor(input_dict)
        
        # 【核心4】计算损失（监督学习目标）
        policy_output['action'] = target_action
        policy_output.update(user_feedback)
        loss_dict = self.actor.get_loss(input_dict, policy_output)
        actor_loss = loss_dict['loss']
        
        # 反向传播与优化
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 记录训练指标
        for k in loss_dict:
            try:
                self.training_history[k].append(loss_dict[k].item())
            except:
                self.training_history[k].append(loss_dict[k])
        
        # 【核心5】在线评估（不用于训练）
        # 在模拟器中评估当前策略，监控训练进度
        self.online_one_step_eval(sample_observation, candidate_info)
    
    def online_one_step_eval(self, observation, candidate_info):
        """
        在线评估：在模拟器中评估策略性能
        
        功能：
            1. 使用当前策略在模拟器中生成推荐
            2. 获取模拟器的用户反馈
            3. 计算在线指标（奖励、多样性等）
            4. 计算离线指标（NDCG、MRR）
        
        注意：
            - 这里不更新Buffer（离线训练不使用Buffer）
            - 这里不更新模型（仅评估）
            - 评估用于监控训练进度，不作为训练信号
        
        Args:
            observation: 环境观察
            candidate_info: 候选物品信息
        """
        # 在线评估
        with torch.no_grad():  # 评估阶段不需要梯度
            self.env.current_observation = observation
            
            # 【步骤1】策略选择动作
            input_dict = {
                'observation': observation,
                'candidates': candidate_info,
                'action_dim': self.env.action_dim,
                'action': None,        # 待生成
                'response': None,      # 待观察
                'epsilon': 0,          # 不探索（贪婪策略）
                'do_explore': False,   # 不探索
                'is_train': False      # 评估模式
            }
            policy_output = self.actor(input_dict)
            
            # 【步骤2】获取模拟器反馈
            action_dict = {'action': policy_output['action']}
            user_feedback = self.env.get_response(action_dict)
            
            # 【步骤3】计算奖励和在线指标
            user_feedback['immediate_response_weight'] = self.immediate_response_weight
            R = self.reward_func(user_feedback).detach()
            user_feedback['reward'] = R
            
            # 记录在线指标
            self.eval_history['avg_reward'].append(R.mean().item())
            self.eval_history['max_reward'].append(R.max().item())
            self.eval_history['reward_variance'].append(torch.var(R).item())
            self.eval_history['coverage'].append(user_feedback['coverage'])
            self.eval_history['intra_slate_diversity'].append(user_feedback['ILD'])
            for i, resp in enumerate(self.env.response_types):
                self.eval_history[f'{resp}_rate'].append(
                    user_feedback['immediate_response'][:, :, i].mean().item()
                )
            
            # 【步骤4】计算离线指标
            # 提取动作和反馈
            target_action = policy_output['action']  # (B, slate_size)
            target_response = user_feedback['immediate_response']  # (B, slate_size, response_dim)
            
            # 计算NDCG和MRR
            metric_out = self.get_offline_metrics(
                observation, 
                candidate_info,
                target_action, 
                target_response
            )
            
            # 记录离线指标
            for k, v in metric_out.items():
                if k in self.eval_history:
                    self.eval_history[k].append(v)
    
    def get_offline_metrics(self, observation, candidate_info, target_action, target_response):
        """
        计算离线评估指标
        
        指标：
            - NDCG (Normalized Discounted Cumulative Gain): 归一化折扣累积增益
            - MRR (Mean Reciprocal Rank): 平均倒数排名
        
        计算方式：
            1. 根据用户反馈计算每个物品的点式奖励
            2. 获取策略对每个物品的选择概率
            3. 按概率排序，计算排序指标
        
        Args:
            observation: 环境观察
            candidate_info: 候选物品
            target_action: 推荐动作 (B, slate_size)
            target_response: 用户反馈 (B, slate_size, response_dim)
        
        Returns:
            metric_dict: 包含NDCG和MRR的字典
        """
        B = target_action.shape[0]
        K = self.env.action_dim  # slate size
        
        # 计算点式奖励
        # 将多种反馈加权求和作为物品的相关度
        # (B, slate_size)
        point_reward = torch.sum(
            target_response * self.immediate_response_weight.view(1, 1, -1),
            dim=2
        ).view(B, K)
        
        # 获取策略概率
        input_dict = {
            'observation': observation,
            'candidates': candidate_info,
            'action_dim': self.env.action_dim,
            'action': target_action,
            'response': None,
            'epsilon': 0,
            'do_explore': False,
            'is_train': False
        }
        policy_output = self.actor(input_dict)
        
        # 策略对每个物品的选择概率 (B, slate_size)
        P = policy_output['prob'].view(B, K).detach()
        
        # 初始化指标字典
        metric_dict = {'NDCG': 0, 'MRR': 0}
        metric_dict.update({f"NDCG_{t}": 0 for t in range(K)})
        metric_dict.update({f"MRR_{t}": 0 for t in range(K)})
        
        # 计算每个位置的排序指标
        NDCG = 0
        MRR = 0
        for t in range(K):
            # 按概率排序
            P_t, P_indices = torch.sort(P[:, t], descending=True)
            
            # 计算该位置的NDCG和MRR
            ranking_metrics = self.get_ranking_metrics(
                P[:, t],              # 该位置的概率分布
                point_reward[:, t],   # 该位置的相关度
                self.NDCG_discount    # NDCG折扣因子
            )
            
            metric_dict[f'NDCG_{t}'] = ranking_metrics['NDCG']
            metric_dict[f'MRR_{t}'] = ranking_metrics['MRR']
            NDCG += ranking_metrics['NDCG']
            MRR += ranking_metrics['MRR']
        
        # 平均所有位置的指标
        metric_dict['NDCG'] = NDCG / K
        metric_dict['MRR'] = MRR / K
        
        return metric_dict
    
    def get_ranking_metrics(self, sorted_P, relevance, NDCG_discount):
        """
        计算排序指标（NDCG和MRR）
        
        NDCG (Normalized Discounted Cumulative Gain):
            DCG = Σ (relevance_i * discount_i)
            IDCG = Σ (sorted_relevance_i * discount_i)
            NDCG = DCG / IDCG
        
        MRR (Mean Reciprocal Rank):
            MRR = Σ (relevance_i / position_i)
        
        Args:
            sorted_P: 排序后的概率
            relevance: 相关度分数
            NDCG_discount: NDCG折扣因子
        
        Returns:
            {'NDCG': scalar, 'MRR': scalar}
        """
        # 计算DCG
        DCG = torch.sum(relevance * NDCG_discount[:len(sorted_P)])
        
        # 计算IDCG（理想DCG）
        # 按相关度排序（理想排序）
        sorted_reward, _ = torch.sort(relevance, descending=True)
        IDCG = torch.sum(sorted_reward * NDCG_discount[:len(sorted_P)])
        
        # 计算NDCG
        NDCG = (DCG / IDCG).item()
        
        # 计算MRR
        MRR = torch.mean(relevance * self.rank_discount[:len(sorted_P)]).item()
        
        return {'NDCG': NDCG, 'MRR': MRR}
    
    def test(self, *episode_args):
        """
        离线测试：在测试集上评估
        
        流程：
            1. 加载测试集数据
            2. 对每个批次计算离线指标
            3. 汇总并保存结果
        
        Args:
            episode_args: (phase,) 其中phase='val'或'test'
        """
        phase = episode_args[0]
        assert phase == 'val' or phase == 'test'
        
        # 初始化测试数据加载器
        self.env.reader.set_phase(phase)
        self.offline_test_iter = DataLoader(
            self.env.reader,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=8
        )
        
        # 初始化测试报告
        K = self.env.action_dim
        test_report = {"NDCG": [], "MRR": []}
        test_report.update({f"NDCG_{t}": [] for t in range(K)})
        test_report.update({f"MRR_{t}": [] for t in range(K)})
        
        # 遍历测试集
        with torch.no_grad():
            # 全物品集作为候选
            candidate_info = self.env.get_candidate_info(None)
            
            for i, test_batch in tqdm(enumerate(self.offline_test_iter)):
                B = test_batch['user_id'].shape[0]
                observation = self.env.get_observation_from_batch(test_batch)
                
                # 提取目标动作和反馈
                target_action = test_batch['item_id'].view(B, K) - 1  # (B, slate_size)
                target_response = torch.cat([
                    test_batch[resp].view(B, K, 1)
                    for i, resp in enumerate(self.response_types)
                ], dim=2)  # (B, slate_size, response_dim)
                
                # 计算指标
                metric_out = self.get_offline_metrics(
                    observation,
                    candidate_info,
                    target_action,
                    target_response
                )
                
                # 累积结果
                for k, v in metric_out.items():
                    if k in test_report:
                        test_report[k].append(v)
        
        # 汇总结果
        test_report = {k: np.mean(v) for k, v in test_report.items()}
        log_str = f"{test_report}"
        print(f"Offline test_result:\n{log_str}")
        
        # 保存测试结果
        with open(self.save_path + "_offline_test.report", 'w') as outfile:
            outfile.write(log_str)
        
        return None

    def save(self):
        """保存模型和优化器状态"""
        torch.save(self.actor.state_dict(), self.save_path + "_actor")
        torch.save(self.actor_optimizer.state_dict(), self.save_path + "_actor_optimizer")

    def load(self):
        """加载模型和优化器状态（用于继续训练）"""
        self.actor.load_state_dict(
            torch.load(self.save_path + "_actor", map_location=self.device, weights_only=False)
        )
        self.actor_optimizer.load_state_dict(
            torch.load(self.save_path + "_actor_optimizer", map_location=self.device, weights_only=False)
        )
        self.actor_target = copy.deepcopy(self.actor)