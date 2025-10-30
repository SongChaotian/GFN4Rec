from matplotlib.pyplot import axes, axis
from model.simulator.KRMBUserResponse import KRMBUserResponse
from model.components import DNN
import torch
import torch.nn as nn

class KRMBUserResponse_MaxOut(KRMBUserResponse):
    '''
    KuaiRand 多行为用户响应模型 - MaxOut版本
    使用多个预测头集成，取最大值作为最终预测
    '''
    
    @staticmethod
    def parse_model_args(parser):
        parser = KRMBUserResponse.parse_model_args(parser)
        parser.add_argument('--n_ensemble', type=int, default=2, help='item encoding size')
        return parser
        
    def __init__(self, args, reader_stats, device):
        self.n_ensemble = args.n_ensemble
        super().__init__(args, reader_stats, device)
        
    def to(self, device):
        return super(KRMBUserResponse_MaxOut, self).to(device)

    def _define_params(self, args):
        stats = self.reader_stats
        self.user_feature_dims = stats['user_feature_dims']
        self.item_feature_dims = stats['item_feature_dims']

        # user embedding
        self.uIDEmb = nn.Embedding(stats['n_user'] + 1, args.user_latent_dim)
        self.uFeatureEmb = {}
        for f, dim in self.user_feature_dims.items():
            embedding_module = nn.Linear(dim, args.user_latent_dim)
            self.add_module(f'UFEmb_{f}', embedding_module)
            self.uFeatureEmb[f] = embedding_module
            
        # item embedding
        self.iIDEmb = nn.Embedding(stats['n_item'] + 1, args.item_latent_dim)
        self.iFeatureEmb = {}
        for f, dim in self.item_feature_dims.items():
            embedding_module = nn.Linear(dim, args.item_latent_dim)
            self.add_module(f'IFEmb_{f}', embedding_module)
            self.iFeatureEmb[f] = embedding_module
        
        # feedback embedding
        self.feedback_types = stats['feedback_type']
        self.feedback_dim = stats['feedback_size']
        self.output_dim = self.feedback_dim * args.n_ensemble  # 多头输出
        self.feedbackEncoder = nn.Linear(self.feedback_dim, args.enc_dim)
        self.set_behavior_hyper_weight(torch.ones(self.feedback_dim))
        
        # item embedding kernel encoder
        self.itemEmbNorm = nn.LayerNorm(args.item_latent_dim)
        self.userEmbNorm = nn.LayerNorm(args.user_latent_dim)
        self.itemFeatureKernel = nn.Linear(args.item_latent_dim, args.enc_dim)
        self.userFeatureKernel = nn.Linear(args.user_latent_dim, args.enc_dim)
        self.encDropout = nn.Dropout(self.dropout_rate)
        self.encNorm = nn.LayerNorm(args.enc_dim)
        
        # positional embedding
        self.max_len = stats['max_seq_len']
        self.posEmb = nn.Embedding(self.max_len, args.enc_dim)
        self.pos_emb_getter = torch.arange(self.max_len, dtype=torch.long)
        self.attn_mask = ~torch.tril(torch.ones((self.max_len, self.max_len), dtype=torch.bool))
        
        # Transformer序列编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=2 * args.enc_dim,
            dim_feedforward=args.transformer_d_forward,
            nhead=args.attn_n_head,
            dropout=args.dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=args.transformer_n_layer)
        self.state_dim = 3 * args.enc_dim
        
        # 状态归一化
        self.stateNorm = nn.LayerNorm(args.enc_dim)
        
        # 行为预测打分器（输出维度扩展为 n_ensemble 倍）
        self.scorer_hidden_dims = args.scorer_hidden_dims
        self.scorer = DNN(
            self.state_dim,
            args.state_hidden_dims,
            self.output_dim * args.enc_dim,
            dropout_rate=args.dropout_rate,
            do_batch_norm=True
        )

    def get_forward(self, feed_dict: dict):
        B = feed_dict['user_id'].shape[0]
        
        # 编码目标物品
        item_enc, item_reg = self.get_item_encoding(
            feed_dict['item_id'],
            {k[3:]: v for k, v in feed_dict.items() if k[:3] == 'if_'},
            B
        )
        item_enc = item_enc.view(B, -1, 1, self.enc_dim)
        
        # 编码用户状态
        state_encoder_output = self.encode_state(feed_dict, B)
        user_state = state_encoder_output['state'].view(B, 1, self.state_dim)

        # 计算行为预测分数（MaxOut机制）
        behavior_scores, point_scores = self.get_pointwise_scores(user_state, item_enc, B)

        # 正则化
        reg = self.get_regularization(
            self.feedbackEncoder,
            self.itemFeatureKernel,
            self.userFeatureKernel,
            self.posEmb,
            self.transformer,
            self.scorer
        )
        reg = reg + state_encoder_output['reg'] + item_reg
        
        return {'preds': behavior_scores, 'state': user_state, 'reg': reg}
    
    def get_pointwise_scores(self, user_state, item_enc, B):
        # 用DNN将用户状态映射为多头注意力权重
        behavior_attn = self.scorer(user_state)
        behavior_attn = self.stateNorm(behavior_attn.view(B, 1, self.output_dim, self.enc_dim))
        
        # 计算用户-物品匹配度（每种行为有 n_ensemble 个预测头）
        point_scores = (behavior_attn * item_enc).mean(dim=-1)
        point_scores = point_scores.view(B, -1, self.feedback_dim, self.n_ensemble)
        
        # MaxOut: 取多个预测头中的最大值
        behavior_scores, max_indices = torch.max(point_scores, dim=-1)
        
        return behavior_scores, point_scores
