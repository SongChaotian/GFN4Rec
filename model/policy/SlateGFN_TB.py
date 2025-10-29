"""
SlateGFN_TB - 基于轨迹平衡的生成流网络列表推荐策略

核心思想：
    - 使用流网络建模推荐列表的生成过程
    - 通过轨迹平衡损失使列表生成概率与奖励成正比: P(O|u) ∝ R(u,O)
    - 自回归生成过程捕捉物品间的相互影响
    
与详细平衡(DB)的区别：
    - TB: 优化整条轨迹，直接匹配完整生成概率与奖励
    - DB: 优化每个生成步骤，分步匹配流值
    - TB适合小规模动作空间，DB适合大规模动作空间
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

from model.general import BaseModel
from model.components import DNN
from model.policy.BaseOnlinePolicy import BaseOnlinePolicy

class SlateGFN_TB(BaseOnlinePolicy):
    """
    基于轨迹平衡(Trajectory Balance)的GFlowNet列表推荐策略
    
    生成流网络框架：
        - 将推荐列表生成建模为在树结构上的概率流
        - 学习使得 P(O|u) ∝ R(u,O)，即生成概率与奖励成正比
        - 通过轨迹平衡损失优化整条生成轨迹
    """
    
    @staticmethod
    def parse_model_args(parser):
        """
        模型参数配置
        
        关键参数：
            - gfn_forward_hidden_dims: 前向策略网络P_F的隐藏层维度
            - gfn_flowzero_hidden_dims: 初始流估计器F_0的隐藏层维度
            - gfn_forward_offset: 前向概率平滑偏移b_f，用于稳定训练
            - gfn_reward_smooth: 奖励平滑偏移b_r，避免除零
            - gfn_Z: 归一化偏移b_z，控制流的全局尺度
        """
        parser = BaseOnlinePolicy.parse_model_args(parser) 
        parser.add_argument('--gfn_forward_hidden_dims', type=int, nargs="+", default=[128], help='前向策略网络的隐藏层维度')
        parser.add_argument('--gfn_flowzero_hidden_dims', type=int, nargs="+", default=[128], help='初始流估计器F_0的隐藏层维度')
        parser.add_argument('--gfn_forward_offset', type=float, default=1.0, help='前向概率平滑偏移b_f，用于稳定训练')
        parser.add_argument('--gfn_reward_smooth', type=float, default=1.0, help='奖励平滑偏移b_r，避免除零')
        parser.add_argument('--gfn_Z', type=float, default=0., help='归一化偏移b_z，控制流的全局尺度')
        
        return parser
        
    def __init__(self, args, reader_stats, device):
        """初始化GFlowNet模型"""
        self.gfn_forward_hidden_dims = args.gfn_forward_hidden_dims
        self.gfn_flowzero_hidden_dims = args.gfn_flowzero_hidden_dims
        self.gfn_forward_offset = args.gfn_forward_offset  # b_f
        self.gfn_reward_smooth = args.gfn_reward_smooth    # b_r
        self.gfn_Z = args.gfn_Z                            # b_z
        super().__init__(args, reader_stats, device)
        self.display_name = "GFN_TB"
        
    def to(self, device):
        """将模型移动到指定设备"""
        new_self = super(SlateGFN_TB, self).to(device)
        return new_self

    def _define_params(self, args):
        """
        定义模型参数
        
        组件：
            1. pForwardEncoder: 前向策略网络 P_F(a_t|u, O_{t-1})
               输入: 用户状态 + 当前已生成列表的编码
               输出: 选择权重向量
               
            2. logFlowZero: 初始流估计器 F_0(u)
               输入: 用户状态
               输出: 初始流值 log F(u, ∅)
               作用: 为每个用户分配个性化的初始流，表示奖励先验
        """
        super()._define_params(args)
        
        # 前向策略网络: P_F(a_t|u, O_{t-1})
        # 输入: [user_state, O_{t-1}] → 输出: selection_weight
        self.pForwardEncoder = DNN(
            self.state_dim + self.enc_dim * self.slate_size, 
            args.gfn_forward_hidden_dims, 
            self.enc_dim, 
            dropout_rate=args.dropout_rate, 
            do_batch_norm=True
        )
        self.pForwardNorm = nn.LayerNorm(self.enc_dim)
        
        # 初始流估计器: F_0(u) = F(u, ∅)
        # 输入: user_state → 输出: log F(u, ∅)
        self.logFlowZero = DNN(
            self.state_dim, 
            args.gfn_flowzero_hidden_dims, 
            1
        )
    
    def generate_action(self, user_state, feed_dict):
        """
        自回归生成推荐列表
        
        生成过程形成K深度的树结构：
            - 每个节点O_t表示部分生成的列表
            - 每条边表示选择的物品a_t
            - 叶节点对应完整的推荐列表O_K
            
        与DB的区别：
            - TB只需要初始流F_0(u)，不需要每个节点的流值F(u,O_t)
            - TB通过轨迹概率∏P_F(a_t)直接连接F_0和奖励R
        
        Args:
            user_state: (B, state_dim) 用户状态编码
            feed_dict: {
                'candidates': 候选物品信息,
                'action_dim': K (推荐列表长度),
                'action': (B, K) 目标列表(训练时使用),
                'do_explore': 是否探索,
                'is_train': 是否训练模式,
                'epsilon': ε-greedy探索率
            }
        
        Returns:
            out_dict: {
                'prob': (B, K) 每步的前向概率P_F(a_t|u,O_{t-1}),
                'logP': (B, K) 每步的对数前向概率log(P_F + b_f),
                'logF0': (B,) 初始流值log F(u, ∅),
                'action': (B, K) 生成的推荐列表,
                'reg': 正则化项
            }
        """
        B = user_state.shape[0]
        candidates = feed_dict['candidates']
        slate_size = feed_dict['action_dim']
        parent_slate = feed_dict['action']  # (B, K)
        do_explore = feed_dict['do_explore']
        is_train = feed_dict['is_train']
        epsilon = feed_dict['epsilon']
        
        # 判断候选集是否按batch组织
        batch_wise = candidates['item_id'].shape[0] != 1
        if is_train:
            assert not batch_wise  # 训练时使用全物品集
        
        # ε-greedy: epsilon概率进行均匀采样
        do_uniform = np.random.random() < epsilon
            
        # 计算初始流值: log F(u, ∅)
        # 表示用户u的奖励先验，个性化不同用户的流分配
        logF0 = self.logFlowZero(user_state)  # (B,)
        
        # 获取候选物品编码 (1, L, enc_dim) 或 (B, L, enc_dim)
        candidate_item_enc, reg = self.userEncoder.get_item_encoding(
            candidates['item_id'], 
            {k[5:]: v for k, v in candidates.items() if k != 'item_id'}, 
            B if batch_wise else 1
        )
        
        # 初始化输出
        current_P = torch.zeros(B, slate_size).to(self.device)      # 前向概率 P_F
        current_logP = torch.zeros(B, slate_size).to(self.device)   # 对数前向概率 log(P_F + b_f)
        current_action = torch.zeros(B, slate_size).to(torch.long).to(self.device)  # 动作
        current_list_emb = torch.zeros(B, slate_size, self.enc_dim).to(self.device)  # 列表编码
        
        # 自回归生成: O_0 → O_1 → ... → O_K
        for i in range(slate_size):
            # 当前状态: s_t = [user_state, O_{t-1}]
            current_state = torch.cat(
                (user_state.view(B, self.state_dim), current_list_emb.view(B, -1)), 
                dim=1
            )
            
            # 计算选择权重 (B, enc_dim)
            selection_weight = self.pForwardEncoder(current_state)
            selection_weight = self.pForwardNorm(selection_weight)
            
            # 计算候选物品得分 (B, L)
            # score = <selection_weight, item_encoding>
            score = torch.sum(
                selection_weight.view(B, 1, self.enc_dim) * candidate_item_enc, 
                dim=-1
            )
            
            # 前向概率: P_F(a_t|u, O_{t-1})
            prob = torch.softmax(score, dim=1)  # (B, L)
            # 对数概率（带平滑）: log(P_F + b_f)
            logP = torch.log(prob + self.gfn_forward_offset)  # (B, L)

            # 根据模式选择动作
            if is_train or torch.is_tensor(parent_slate):
                # 训练模式: 使用目标动作计算概率(teacher forcing)
                action_at_i = parent_slate[:, i].long()
                current_P[:, i] = torch.gather(prob, 1, action_at_i.view(-1, 1)).view(-1)
                current_logP[:, i] = torch.gather(logP, 1, action_at_i.view(-1, 1)).view(-1)
                current_list_emb[:, i, :] = candidate_item_enc.view(-1, self.enc_dim)[action_at_i]
                current_action[:, i] = action_at_i
            else:
                # 推理模式: 采样或贪心选择
                if i > 0:
                    # 移除已选择的物品(避免重复)
                    prob.scatter_(1, current_action[:, :i], 0)
                
                if do_explore:
                    # 探索: 分类采样或均匀采样
                    if do_uniform:
                        indices = Categorical(torch.ones_like(prob)).sample()
                    else:
                        indices = Categorical(prob).sample()
                else: 
                    # 利用: 贪心选择
                    _, indices = torch.topk(prob, k=1, dim=1)
                
                indices = indices.view(-1).detach().long()
                current_action[:, i] = indices
                current_logP[:, i] = torch.gather(logP, 1, indices.view(-1, 1)).view(-1)
                current_P[:, i] = torch.gather(prob, 1, indices.view(-1, 1)).view(-1)

                # 更新列表编码
                if batch_wise:
                    for j in range(B):
                        current_list_emb[j, i, :] = candidate_item_enc[j, indices[j]]
                else:
                    current_list_emb[:, i, :] = candidate_item_enc.view(-1, self.enc_dim)[indices]
        
        # 计算正则化
        if is_train:
            reg = self.get_regularization(self.logFlowZero, self.pForwardEncoder)
        else:
            reg = 0

        out_dict = {
            'logP': current_logP,    # log(P_F + b_f)
            'prob': current_P,       # P_F
            'action': current_action, 
            'logF0': logF0,          # log F(u, ∅)
            'reg': reg
        }
        return out_dict
    
    def get_loss(self, feed_dict, out_dict):
        """
        计算轨迹平衡损失(Trajectory Balance Loss)
        
        论文公式(7)：
            L_TB = (log b_z + log F_0(u) + Σ_t log(P_F(a_t|u,O_{t-1}) + b_f) - log(R(u,O_K) + b_r))²
        
        简化为:
            L_TB = (log b_z + log F_0(u) + Σ_t log(P_F(a_t) + b_f) - log(R(u,O) + b_r))²
        
        物理意义：
            - 前向部分: log b_z + log F_0(u) + Σ log P_F(a_t)
              表示通过策略生成轨迹的对数流量
            - 后向部分: log(R(u,O) + b_r)
              表示轨迹终点的对数奖励
            - TB损失最小化两者差异，使得生成概率与奖励成正比
        
        与DB的区别：
            - TB: 优化整条轨迹，损失函数只计算一次
            - DB: 优化每个步骤，损失函数计算K+1次(K个中间节点+1个叶节点)
            - TB方差较大但更直接，DB方差较小但更复杂
        
        Args:
            feed_dict: 输入字典
            out_dict: {
                'state': (B, state_dim),
                'logP': (B, K) 对数前向概率log(P_F + b_f),
                'logF0': (B,) 初始流值log F(u, ∅),
                'action': (B, K),
                'reg': 正则化项,
                'immediate_response': (B, K*n_feedback),
                'reward': (B,) 列表奖励R(u,O)
            }
        
        Returns:
            loss_dict: 包含损失的字典
        """
        # 【前向部分】轨迹的对数流量
        # log b_z + log F_0(u) + Σ_t log(P_F(a_t) + b_f)
        # (B,)
        forward_part = out_dict['logF0'].view(-1) + self.gfn_Z
        forward_part = forward_part + torch.sum(out_dict['logP'], dim=1)
        
        # 【后向部分】轨迹终点的对数奖励
        # log(R(u,O) + b_r)
        # (B,)
        backward_part = torch.log(out_dict['reward'] + self.gfn_reward_smooth).view(-1)
        
        # 【轨迹平衡损失】
        # L_TB = (forward_part - backward_part)²
        # 最小化生成流与奖励之间的差异
        TB_loss = torch.mean((forward_part - backward_part).pow(2))
        
        # 总损失 = TB损失 + L2正则
        loss = TB_loss + self.l2_coef * out_dict['reg']
        
        return {
            'loss': loss, 
            'TB_loss': TB_loss
        }

    def get_loss_observation(self):
        """返回需要记录的损失指标名称"""
        return ['loss', 'TB_loss']