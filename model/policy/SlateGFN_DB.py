import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

from model.general import BaseModel
from model.components import DNN
from model.policy.BaseOnlinePolicy import BaseOnlinePolicy

class SlateGFN_DB(BaseOnlinePolicy):
    '''
    基于详细平衡(Detailed Balance)的生成流网络(GFlowNet)列表推荐策略
    
    核心思想：
    - 使用流网络建模推荐列表的生成过程
    - 通过详细平衡损失使列表生成概率与奖励成正比: P(O|u) ∝ R(u,O)
    - 自回归生成过程捕捉物品间的相互影响
    '''
    
    @staticmethod
    def parse_model_args(parser):
        '''
        模型参数配置
        - gfn_forward_hidden_dims: 前向策略网络P_F的隐藏层维度
        - gfn_flow_hidden_dims: 流估计器F的隐藏层维度
        - gfn_forward_offset: 前向概率平滑偏移b_f (论文中的offset)
        - gfn_reward_smooth: 奖励平滑偏移b_r
        - gfn_Z: 归一化偏移b_z (对应论文中的log partition function)
        '''
        parser = BaseOnlinePolicy.parse_model_args(parser) 
        parser.add_argument('--gfn_forward_hidden_dims', type=int, nargs="+", default=[128], help='前向策略网络的隐藏层维度')
        parser.add_argument('--gfn_flow_hidden_dims', type=int, nargs="+", default=[128], help='流估计器的隐藏层维度')
        parser.add_argument('--gfn_forward_offset', type=float, default=1.0, help='前向概率平滑偏移b_f，用于稳定训练')
        parser.add_argument('--gfn_reward_smooth', type=float, default=1.0, help='奖励平滑偏移b_r，避免除零')
        parser.add_argument('--gfn_Z', type=float, default=0., help='归一化偏移b_z，控制流的全局尺度')
        
        return parser
        
    def __init__(self, args, reader_stats, device):
        '''初始化GFlowNet模型'''
        self.gfn_forward_hidden_dims = args.gfn_forward_hidden_dims
        self.gfn_flow_hidden_dims = args.gfn_flow_hidden_dims
        self.gfn_forward_offset = args.gfn_forward_offset  # b_f
        self.gfn_reward_smooth = args.gfn_reward_smooth    # b_r
        self.gfn_Z = args.gfn_Z                            # b_z
        super().__init__(args, reader_stats, device)
        self.display_name = "GFN_DB"
        
    def to(self, device):
        '''将模型移动到指定设备'''
        new_self = super(SlateGFN_DB, self).to(device)
        return new_self

    def _define_params(self, args):
        '''定义模型参数'''
        super()._define_params(args)
        
        # 前向策略网络: P_F(a_t|u, O_{t-1})
        # 输入: 用户状态 + 当前已生成列表的编码
        self.pForwardEncoder = DNN(
            self.state_dim + self.enc_dim * self.slate_size, 
            args.gfn_forward_hidden_dims, 
            self.enc_dim, 
            dropout_rate=args.dropout_rate, 
            do_batch_norm=True
        )
        self.pForwardNorm = nn.LayerNorm(self.enc_dim)
        
        # 流估计器: F_φ(u, O_t)
        # 估计每个状态节点的流值
        self.logFlow = DNN(
            self.state_dim + self.enc_dim * self.slate_size, 
            args.gfn_flow_hidden_dims, 
            1, 
            dropout_rate=args.dropout_rate, 
            do_batch_norm=True
        )

    def generate_action(self, user_state, feed_dict):
        '''
        自回归生成推荐列表
        
        生成过程形成K深度的树结构:
        - 每个节点O_t表示部分生成的列表
        - 每条边表示选择的物品a_t
        - 叶节点对应完整的推荐列表O_K
        
        @input:
        - user_state: (B, state_dim) 用户状态编码
        - feed_dict: {
            'candidates': 候选物品信息,
            'action_dim': K (推荐列表长度),
            'action': (B, K) 目标列表(训练时使用),
            'do_explore': 是否探索,
            'is_train': 是否训练模式,
            'epsilon': ε-greedy探索率
          }
        @output:
        - out_dict: {
            'prob': (B, K) 每步的前向概率P_F(a_t|u,O_{t-1}),
            'logF': (B, K+1) 每步的流值log F(u,O_t),
            'action': (B, K) 生成的推荐列表,
            'reg': 正则化项
          }
        '''
        B = user_state.shape[0]
        candidates = feed_dict['candidates']
        slate_size = feed_dict['action_dim']
        parent_slate = feed_dict['action']
        do_explore = feed_dict['do_explore']
        is_train = feed_dict['is_train']
        epsilon = feed_dict['epsilon']
        
        # 判断候选集是否按batch组织
        batch_wise = candidates['item_id'].shape[0] != 1
        if is_train:
            assert not batch_wise  # 训练时使用全物品集
        
        # ε-greedy: epsilon概率进行均匀采样
        do_uniform = np.random.random() < epsilon
        
        # 获取候选物品编码 (1, L, enc_dim) 或 (B, L, enc_dim)
        candidate_item_enc, reg = self.userEncoder.get_item_encoding(
            candidates['item_id'], 
            {k[5:]: v for k, v in candidates.items() if k != 'item_id'}, 
            B if batch_wise else 1
        )
        
        # 初始化输出
        current_P = torch.zeros(B, slate_size).to(self.device)      # 前向概率
        current_action = torch.zeros(B, slate_size).to(torch.long).to(self.device)  # 动作
        current_list_emb = torch.zeros(B, slate_size, self.enc_dim).to(self.device)  # 列表编码
        current_flow = torch.zeros(B, slate_size + 1).to(self.device)  # 流值(包含初始和终止状态)
        
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
            prob = torch.softmax(score, dim=1)
            
            if is_train or torch.is_tensor(parent_slate):
                # 训练模式: 使用目标动作计算概率(teacher forcing)
                action_at_i = parent_slate[:, i]
                current_P[:, i] = torch.gather(prob, 1, action_at_i.view(-1, 1)).view(-1)
                current_list_emb[:, i, :] = candidate_item_enc.view(-1, self.enc_dim)[action_at_i]
                current_flow[:, i] = self.logFlow(current_state).view(-1)
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
                
                indices = indices.view(-1).detach()
                current_action[:, i] = indices
                current_P[:, i] = torch.gather(prob, 1, indices.view(-1, 1)).view(-1)
                
                # 更新列表编码
                if batch_wise:
                    for j in range(B):
                        current_list_emb[j, i, :] = candidate_item_enc[j, indices[j]]
                else:
                    current_list_emb[:, i, :] = candidate_item_enc.view(-1, self.enc_dim)[indices]
        
        if is_train:
            # 计算终止状态O_K的流值
            current_state = torch.cat(
                (user_state.view(B, self.state_dim), current_list_emb.view(B, -1)), 
                dim=1
            )
            current_flow[:, -1] = self.logFlow(current_state).view(-1)
            reg = self.get_regularization(self.logFlow, self.pForwardEncoder)
        else:
            reg = 0

        out_dict = {
            'prob': current_P, 
            'action': current_action, 
            'logF': current_flow, 
            'reg': reg
        }
        return out_dict
    
    def get_loss(self, feed_dict, out_dict):
        '''
        计算详细平衡损失(Detailed Balance Loss)
        
        论文公式(8)和(9):
        对于叶节点: L_DB = (log F(u,O_K) - log(R(u,O_K) + b_r))²
        对于中间节点: L_DB = (log(b_z/K) + log(F(u,O_{t-1}) * P_F(a_t|u,O_{t-1})) - log F(u,O_t))²
        
        由于树结构中P_B(O_{t-1}|u,O_t) = 1，简化为:
        L_DB = (log(b_z/K) + log F(u,O_{t-1}) + log(P_F(a_t|u,O_{t-1}) + b_f) - log F(u,O_t))²
        
        @input:
        - feed_dict: 输入字典
        - out_dict: {
            'state': (B, state_dim),
            'prob': (B, K) 前向概率P_F,
            'logF': (B, K+1) 流值F,
            'action': (B, K),
            'reg': 正则化项,
            'immediate_response': (B, K*n_feedback),
            'reward': (B,) 列表奖励R(u,O)
          }
        @output:
        - loss_dict: 包含各项损失的字典
        '''
        # 父节点流值: log F(u, O_{t-1}), (B, K)
        parent_flow = out_dict['logF'][:, :-1]
        # 当前节点流值: log F(u, O_t), (B, K)
        current_flow = out_dict['logF'][:, 1:]
        # 前向对数概率: log(P_F(a_t|u,O_{t-1}) + b_f), (B, K)
        log_P = torch.log(out_dict['prob'] + self.gfn_forward_offset)
        
        # 前向部分: log F(u,O_{t-1}) + log P_F + b_z/K
        forward_part = parent_flow + log_P + self.gfn_Z
        # 后向部分: log F(u,O_t)
        backward_part = current_flow
        
        # 详细平衡损失(中间节点): Σ(forward - backward)²
        DB_loss = torch.mean((forward_part - backward_part).pow(2))
        
        # 终止状态损失(叶节点): (log F(u,O_K) + b_z - log(R(u,O_K) + b_r))²
        terminal_loss = (
            current_flow[:, -1] + self.gfn_Z 
            - torch.log(out_dict['reward'] + self.gfn_reward_smooth + 1e-6).view(-1)
        ).pow(2)
        terminal_loss = torch.mean(terminal_loss)
        
        # 总损失 = DB损失 + 终止损失 + L2正则
        loss = DB_loss + terminal_loss + self.l2_coef * out_dict['reg']
        
        return {
            'loss': loss, 
            'DB_loss': DB_loss, 
            'terminal_loss': terminal_loss, 
            'forward_part': torch.mean(forward_part), 
            'backward_part': torch.mean(backward_part), 
            'prob': torch.mean(out_dict['prob'])
        }

    def get_loss_observation(self):
        '''返回需要记录的损失指标名称'''
        return ['loss', 'DB_loss', 'terminal_loss', 'forward_part', 'backward_part', 'prob']
