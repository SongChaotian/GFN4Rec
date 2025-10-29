"""
BaseOnlineAgent - 在线强化学习Agent

功能：通过与环境实时交互学习推荐策略
核心特点：
    1. 在线交互：与模拟环境实时交互产生训练数据
    2. 经验回放：使用Buffer存储和采样交互经验
    3. 探索策略：ε-greedy探索平衡探索与利用
    4. 增量学习：边交互边学习
"""

import time
import copy
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import utils
from model.agent.reward_func import *


class BaseOnlineAgent():
    """
    在线强化学习Agent基类
    
    训练流程：
        1. action_before_train(): 初始化Buffer，随机探索填充初始数据
        2. 主循环: for i in range(n_iter):
            a) run_episode_step(): 与环境交互产生新数据
            b) step_train(): 从Buffer采样批次数据训练模型
        3. action_after_train(): 清理资源
    """
    
    @staticmethod
    def parse_model_args(parser):
        """
        解析命令行参数
        
        关键参数：
            - n_iter: 训练迭代次数
            - start_train_at_step: 随机探索步数（填充Buffer）
            - initial/final_greedy_epsilon: ε-greedy的起始/终止探索率
            - train_every_n_step: 每N步交互进行一次训练
            - batch_size: 从Buffer采样的批次大小
        """
        parser.add_argument('--n_iter', type=int, nargs='+', default=[2000], 
                            help='训练迭代次数')
        parser.add_argument('--train_every_n_step', type=int, default=1, 
                            help='每N次episode采样进行一次训练步')
        parser.add_argument('--reward_func', type=str, default='get_immediate_reward', 
                            help='奖励函数名称（见model.agent.reward_func）')
        parser.add_argument('--single_response', action='store_true', 
                            help='是否只使用单一反馈类型计算奖励')
        parser.add_argument('--start_train_at_step', type=int, default=1000,
                            help='开始训练前的随机探索步数（用于填充Buffer）')
        parser.add_argument('--initial_greedy_epsilon', type=float, default=0.6, 
                            help='ε-greedy的初始探索率')
        parser.add_argument('--final_greedy_epsilon', type=float, default=0.05, 
                            help='ε-greedy的最终探索率')
        parser.add_argument('--elbow_greedy', type=float, default=0.5, 
                            help='探索率衰减的拐点位置（相对于总迭代次数的比例）')
        parser.add_argument('--check_episode', type=int, default=100, 
                            help='每N次迭代记录一次日志和评估指标')
        parser.add_argument('--test_episode', type=int, default=1000, 
                            help='每N次迭代进行一次测试')
        parser.add_argument('--save_episode', type=int, default=1000, 
                            help='每N次迭代保存一次模型')
        parser.add_argument('--save_path', type=str, required=True, 
                            help='模型保存路径')
        parser.add_argument('--batch_size', type=int, default=64, 
                            help='训练批次大小')
        parser.add_argument('--actor_lr', type=float, default=1e-4, 
                            help='Actor网络学习率')
        parser.add_argument('--actor_decay', type=float, default=1e-4, 
                            help='Actor网络权重衰减（L2正则化）')
        parser.add_argument('--explore_rate', type=float, default=1, 
                            help='触发探索的概率')
        return parser
    
    def __init__(self, *input_args):
        """
        初始化Agent
        
        Args:
            input_args: (args, actor, env, buffer)
                - args: 命令行参数
                - actor: 策略网络
                - env: 模拟环境
                - buffer: 经验回放Buffer
        """
        args, actor, env, buffer = input_args
        
        # 基础配置
        self.device = args.device
        self.n_iter = [0] + args.n_iter  # [0, n_iter] 用于支持继续训练
        self.train_every_n_step = args.train_every_n_step
        self.start_train_at_step = args.start_train_at_step
        self.check_episode = args.check_episode
        self.test_episode = args.test_episode
        self.save_episode = args.save_episode
        self.save_path = args.save_path
        self.episode_batch_size = args.episode_batch_size
        self.batch_size = args.batch_size
        self.actor_lr = args.actor_lr
        self.actor_decay = args.actor_decay
        self.reward_func = eval(args.reward_func)
        self.single_response = args.single_response
        self.explore_rate = args.explore_rate
        
        # 环境配置
        self.env = env
        
        # 设置反馈权重：单一反馈 vs 多反馈加权
        if self.single_response:
            self.immediate_response_weight = torch.zeros(len(self.env.response_types)).to(self.device)
            self.immediate_response_weight[0] = 1  # 只使用第一种反馈
        else:
            self.immediate_response_weight = torch.FloatTensor(self.env.response_weights).to(self.device)

        # 探索策略配置
        # 线性探索率衰减器：从initial_epsilon到final_epsilon
        self.exploration_scheduler = utils.LinearScheduler(
            int(sum(args.n_iter) * args.elbow_greedy),  # 衰减持续时长
            args.final_greedy_epsilon,                   # 最终探索率
            initial_p=args.initial_greedy_epsilon        # 初始探索率
        )
        
        # 策略网络配置
        self.actor = actor
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), 
            lr=args.actor_lr, 
            weight_decay=args.actor_decay
        )
        
        # Buffer配置
        self.buffer = buffer
        
        # 日志文件初始化
        if len(self.n_iter) == 2:  # 首次训练
            with open(self.save_path + ".report", 'w') as outfile:
                outfile.write(f"{args}\n")
            with open(self.save_path + "_test.report", 'w') as outfile:
                outfile.write(f"{args}\n")
    
    def train(self):
        """
        【主训练循环】
        
        流程：
            1. action_before_train(): 随机探索填充Buffer
            2. 主循环：
                a) run_episode_step(): 与环境交互
                b) step_train(): 从Buffer采样训练
                c) 定期记录日志、测试、保存模型
            3. action_after_train(): 清理资源
        """
        # 如果是继续训练，加载之前的模型
        if len(self.n_iter) > 2:
            self.load()
        
        t = time.time()
        print("Run procedures before training")
        observation = self.action_before_train()  # 初始化并填充Buffer
        t = time.time()
        start_time = t
        
        # 主训练循环
        print("Training:")
        step_offset = sum(self.n_iter[:-1])
        for i in tqdm(range(step_offset, step_offset + self.n_iter[-1])):
            # 控制标志
            do_buffer_update = True  # 将交互数据存入Buffer
            do_explore = True        # 启用探索

            # 【核心1】在线交互：与环境实时交互产生数据
            observation = self.run_episode_step(
                i,
                self.exploration_scheduler.value(i),  # 动态epsilon：随训练进度递减
                observation, 
                do_buffer_update,  # 将交互经验存入Buffer
                do_explore         # 开启ε-greedy探索
            )

            # 【核心2】训练：从Buffer采样训练策略网络
            if i % self.train_every_n_step == 0:
                self.step_train()

            # 日志记录
            if i % self.check_episode == 0 and i >= self.check_episode:
                t_prime = time.time()
                print(f"Episode step {i}, time diff {t_prime - t}, total time diff {t - start_time})")
                episode_report, train_report = self.get_report()
                log_str = f"step: {i} @ online episode: {episode_report} @ training: {train_report}\n"
                with open(self.save_path + ".report", 'a') as outfile:
                    outfile.write(log_str)
                print(log_str)
                t = t_prime
            
            # 保存模型
            if i % self.save_episode == 0:
                self.save()
            
            # 测试
            if i % self.test_episode == 0:
                observation = self.test(i, observation)
        
        # 训练结束清理
        self.action_after_train()
    
    def action_before_train(self):
        """
        训练前准备
        
        功能：
            1. 重置Buffer
            2. 初始化评估指标记录
            3. 【关键】随机探索填充Buffer初始数据
        
        Returns:
            observation: 环境观察，用于开始训练循环
        """
        # Buffer设置
        self.buffer.reset(self.env, self.actor)
        
        # 训练记录初始化
        self.training_history = {}  # 训练损失等指标
        self.eval_history = {
            'avg_reward': [],           # 平均奖励
            'max_reward': [],           # 最大奖励
            'reward_variance': [],      # 奖励方差
            'coverage': [],             # 覆盖率
            'intra_slate_diversity': [] # 推荐列表内多样性
        }
        # 为每种用户反馈类型添加记录
        self.eval_history.update({f'{resp}_rate': [] for resp in self.env.response_types})
        self.initialize_training_history()
        
        # 【关键】随机探索填充Buffer
        # 使用完全随机策略收集初始经验，避免冷启动问题
        initial_epsilon = 1.0  # 完全随机探索
        observation = self.env.reset()
        
        print(f"Filling buffer with random exploration ({self.start_train_at_step} steps)...")
        for i in tqdm(range(self.start_train_at_step)):
            do_buffer_update = True  # 存入Buffer
            do_explore = np.random.random() < self.explore_rate  # 以explore_rate概率探索
            observation = self.run_episode_step(
                0,                  # episode_iter (填充阶段不重要)
                initial_epsilon,    # epsilon=1.0 完全随机
                observation, 
                do_buffer_update,   # 存入Buffer
                do_explore          # 探索
            )
        
        return observation
    
    def run_episode_step(self, episode_iter, epsilon, observation, 
                        do_buffer_update, do_explore):
        """
        【核心】在线交互：执行一步episode
        
        流程：
            1. 策略选择动作（带探索）
            2. 环境执行动作，返回用户反馈
            3. 计算奖励
            4. 【关键】将经验存入Buffer
            5. 记录评估指标
        
        Args:
            episode_iter: 当前迭代次数
            epsilon: 探索率（ε-greedy）
            observation: 当前环境观察
            do_buffer_update: 是否将经验存入Buffer
            do_explore: 是否启用探索
        
        Returns:
            new_observation: 下一状态的观察
        """
        with torch.no_grad():  # 交互阶段不需要梯度
            # 准备输入
            observation['batch_size'] = self.episode_batch_size
            candidate_info = self.env.get_candidate_info(observation)
            
            # 【步骤1】策略选择动作
            input_dict = {
                'observation': observation,      # 用户状态（画像、历史等）
                'candidates': candidate_info,    # 候选物品信息
                'action_dim': self.env.action_dim,
                'action': None,                  # 待生成
                'response': None,                # 待观察
                'epsilon': epsilon,              # 探索率：控制随机探索 vs 贪婪选择
                'do_explore': do_explore,        # 是否启用探索
                'is_train': False                # 交互阶段，非训练模式
            }
            # 策略网络输出动作（考虑探索）
            policy_output = self.actor(input_dict)
            
            # 【步骤2】环境执行动作
            # 将动作应用到环境，获取用户反馈
            action_dict = {'action': policy_output['action']}
            new_observation, user_feedback, updated_observation = self.env.step(action_dict)
            
            # 【步骤3】计算奖励
            user_feedback['immediate_response_weight'] = self.immediate_response_weight
            R = self.reward_func(user_feedback).detach()  # 根据用户反馈计算奖励
            user_feedback['reward'] = R
            
            # 记录评估指标
            self.eval_history['avg_reward'].append(R.mean().item())
            self.eval_history['max_reward'].append(R.max().item())
            self.eval_history['reward_variance'].append(torch.var(R).item())
            self.eval_history['coverage'].append(user_feedback['coverage'])
            self.eval_history['intra_slate_diversity'].append(user_feedback['ILD'])
            for i, resp in enumerate(self.env.response_types):
                self.eval_history[f'{resp}_rate'].append(
                    user_feedback['immediate_response'][:, :, i].mean().item()
                )
            
            # 【步骤4】【关键】存入Buffer
            # 将交互经验 (s, a, r, s') 存入经验回放Buffer
            if do_buffer_update:
                self.buffer.update(
                    observation,         # 当前状态 s
                    policy_output,       # 动作和策略输出 a, π(a|s)
                    user_feedback,       # 用户反馈和奖励 r
                    updated_observation  # 下一状态 s'
                )
        
        return new_observation
    
    def step_train(self):
        """
        【核心】训练步：从Buffer采样训练
        
        流程：
            1. 从Buffer采样批次数据 (s, a, r, s')
            2. 策略网络前向传播
            3. 计算损失（GFN损失或其他）
            4. 反向传播更新参数
        
        说明：
            - 这里的数据来自Buffer，即之前的在线交互经验
            - 采用off-policy学习：训练数据可能来自不同的策略
        """
        # 【核心】从Buffer采样批次数据
        # 采样格式：(observation, policy_output, user_feedback, next_observation)
        observation, target_output, target_response, _, __ = self.buffer.sample(self.batch_size)
        
        # 前向传播
        observation['batch_size'] = self.episode_batch_size
        candidate_info = self.env.get_candidate_info(observation)
        
        input_dict = {
            'observation': observation,
            'candidates': candidate_info,
            'action_dim': self.env.action_dim,
            'action': target_output['action'],    # Buffer中存储的动作
            'response': target_response,          # Buffer中存储的用户反馈
            'epsilon': 0,                         # 训练时不探索
            'do_explore': False,                  # 训练时不探索
            'is_train': True                      # 训练模式
        }
        policy_output = self.actor(input_dict)

        # 计算损失
        policy_output['action'] = target_output['action']
        policy_output.update(target_response)
        policy_output.update({'immediate_response_weight': self.immediate_response_weight})

        # 调用策略网络的损失函数（如GFN的DB损失或TB损失）
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

        return {"step_loss": self.training_history['loss'][-1]}
    
    def initialize_training_history(self):
        """
        初始化训练历史记录
        根据策略网络定义的损失项初始化记录字典
        """
        self.training_history = {k: [] for k in self.actor.get_loss_observation()}
    
    def get_report(self):
        """
        生成训练报告
        
        Returns:
            episode_report: 在线交互指标（奖励、多样性等）
            train_report: 训练指标（损失等）
        """
        episode_report = {
            k: np.mean(v[-self.check_episode:]) 
            for k, v in self.eval_history.items()
        }
        train_report = {
            k: np.mean(v[-self.check_episode:]) 
            for k, v in self.training_history.items()
        }
        return episode_report, train_report
    
    def action_after_train(self):
        """训练结束后清理"""
        self.env.stop()
    
    def test(self, *episode_args):
        """
        测试：评估当前策略性能
        
        与训练交互类似，但不更新Buffer，不探索
        
        Args:
            episode_args: (episode_iter, observation)
        
        Returns:
            new_observation: 下一状态
        """
        episode_iter, observation = episode_args
        test_report = {}
        
        with torch.no_grad():
            # 准备输入
            observation['batch_size'] = self.episode_batch_size
            candidate_info = self.env.get_candidate_info(observation)
            
            # 策略选择动作（贪婪，不探索）
            input_dict = {
                'observation': observation,
                'candidates': candidate_info,
                'action_dim': self.env.action_dim,
                'action': None,
                'response': None,
                'epsilon': 0,          # 不探索
                'do_explore': False,   # 不探索
                'is_train': False
            }
            policy_output = self.actor(input_dict)
            
            # 环境执行
            action_dict = {'action': policy_output['action']}
            new_observation, user_feedback, updated_observation = self.env.step(action_dict)
            
            # 计算测试指标
            user_feedback['immediate_response_weight'] = self.immediate_response_weight
            R = self.reward_func(user_feedback).detach()
            user_feedback['reward'] = R
            test_report['avg_reward'] = R.mean().item()
            test_report['max_reward'] = R.max().item()
            test_report['reward_variance'] = torch.var(R).item()
            test_report['coverage'] = user_feedback['coverage']
            test_report['intra_slate_diversity'] = user_feedback['ILD']
            for j, resp in enumerate(self.env.response_types):
                test_report[f'{resp}_rate'] = user_feedback['immediate_response'][:, :, j].mean().item()
        
        # 记录测试日志
        train_report = {k: np.mean(v[-self.check_episode:]) for k, v in self.training_history.items()}
        log_str = f"step: {episode_iter} @ online episode: {test_report} @ training: {train_report}\n"
        with open(self.save_path + "_test.report", 'a') as outfile:
            outfile.write(log_str)
        
        return new_observation

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