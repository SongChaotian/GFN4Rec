from matplotlib.pyplot import axes, axis
from model.general import BaseModel
from model.components import DNN
import torch
import torch.nn as nn

class KRMBUserResponse(BaseModel):
    '''
    KuaiRand 多行为用户响应模型
    预测用户对物品的多种行为反馈（点击、点赞、长时观看等）
    '''
    
    @staticmethod
    def parse_model_args(parser):
        parser = BaseModel.parse_model_args(parser)
        
        parser.add_argument('--user_latent_dim', type=int, default=16, help='user latent embedding size')
        parser.add_argument('--item_latent_dim', type=int, default=16, help='item latent embedding size')
        parser.add_argument('--enc_dim', type=int, default=32, help='item encoding size')
        parser.add_argument('--attn_n_head', type=int, default=4, help='number of attention heads in transformer')
        parser.add_argument('--transformer_d_forward', type=int, default=64, help='forward layer dimension in transformer')
        parser.add_argument('--transformer_n_layer', type=int, default=2, help='number of encoder layers in transformer')
        parser.add_argument('--state_hidden_dims', type=int, nargs='+', default=[128],  help='hidden dimensions')
        parser.add_argument('--scorer_hidden_dims', type=int, nargs='+', default=[128], help='hidden dimensions')
        parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropout rate in deep layers')
        return parser
        
    def __init__(self, args, reader_stats, device):
        self.user_latent_dim = args.user_latent_dim
        self.item_latent_dim = args.item_latent_dim
        self.enc_dim = args.enc_dim
        self.attn_n_head = args.attn_n_head
        self.scorer_hidden_dims = args.scorer_hidden_dims
        self.dropout_rate = args.dropout_rate
        super().__init__(args, reader_stats, device)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.state_dim = 3 * args.enc_dim
        
    def to(self, device):
        new_self = super(KRMBUserResponse, self).to(device)
        new_self.attn_mask = new_self.attn_mask.to(device)
        new_self.pos_emb_getter = new_self.pos_emb_getter.to(device)
        new_self.behavior_weight = new_self.behavior_weight.to(device)
        return new_self

    def _define_params(self, args):
        stats = self.reader_stats
        self.user_feature_dims = stats['user_feature_dims']
        self.item_feature_dims = stats['item_feature_dims']

        # user embedding
        self.uIDEmb = nn.Embedding(stats['n_user']+1, args.user_latent_dim)
        self.uFeatureEmb = {}
        for f, dim in self.user_feature_dims.items():
            embedding_module = nn.Linear(dim, args.user_latent_dim)
            self.add_module(f'UFEmb_{f}', embedding_module)
            self.uFeatureEmb[f] = embedding_module
            
        # item embedding
        self.iIDEmb = nn.Embedding(stats['n_item']+1, args.item_latent_dim)
        self.iFeatureEmb = {}
        for f, dim in self.item_feature_dims.items():
            embedding_module = nn.Linear(dim, args.item_latent_dim)
            self.add_module(f'IFEmb_{f}', embedding_module)
            self.iFeatureEmb[f] = embedding_module
        
        # feedback embedding
        self.feedback_types = stats['feedback_type']
        self.feedback_dim = stats['feedback_size']
        self.xtr_dim = 2 * self.feedback_dim
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
        
        # DNN state encoder
        self.stateNorm = nn.LayerNorm(args.enc_dim)
        
        # 行为预测打分器
        self.scorer = DNN(
            3 * args.enc_dim,
            args.state_hidden_dims,
            self.feedback_dim * args.enc_dim,
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
        
        # 编码用户状态
        state_encoder_output = self.encode_state(feed_dict, B)
        user_state = state_encoder_output['state'].view(B, 1, 3 * self.enc_dim)

        # 计算行为预测分数
        behavior_attn = self.scorer(user_state)
        behavior_attn = self.stateNorm(behavior_attn.view(B, 1, self.feedback_dim, self.enc_dim))
        behavior_scores = (behavior_attn * item_enc.view(B, -1, 1, self.enc_dim)).mean(dim=-1)

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
    
    def encode_state(self, feed_dict, B):
        # 编码历史物品序列
        history_enc, history_reg = self.get_item_encoding(
            feed_dict['history'],
            {f: feed_dict[f'history_if_{f}'] for f in self.iFeatureEmb},
            B
        )
        history_enc = history_enc.view(B, self.max_len, self.enc_dim)
        
        # 添加位置编码
        pos_emb = self.posEmb(self.pos_emb_getter).view(1, self.max_len, self.enc_dim)
        seq_enc_feat = self.encNorm(self.encDropout(history_enc + pos_emb))
        
        # 编码历史反馈行为
        feedback_emb = self.get_response_embedding(feed_dict, B)
        seq_enc = torch.cat((seq_enc_feat, feedback_emb), dim=-1)
        
        # Transformer编码序列
        output_seq = self.transformer(seq_enc, mask=self.attn_mask)
        hist_enc = output_seq[:, -1, :].view(B, 2 * self.enc_dim)
        
        # 编码用户画像
        user_enc, user_reg = self.get_user_encoding(
            feed_dict['user_id'],
            {k[3:]: v for k, v in feed_dict.items() if k[:3] == 'uf_'},
            B
        )
        user_enc = self.encNorm(self.encDropout(user_enc)).view(B, self.enc_dim)
        
        # 融合历史和画像
        state = torch.cat([hist_enc, user_enc], dim=1)
        
        return {'output_seq': output_seq, 'state': state, 'reg': user_reg + history_reg}
    
    def get_user_encoding(self, user_ids, user_features, B):
        # 用户ID嵌入
        user_id_emb = self.uIDEmb(user_ids).view(B, 1, self.user_latent_dim)
        
        # 收集所有用户特征嵌入
        user_feature_emb = [user_id_emb]
        for f, fEmbModule in self.uFeatureEmb.items():
            user_feature_emb.append(fEmbModule(user_features[f]).view(B, 1, self.user_latent_dim))
        
        # 合并并编码
        combined_user_emb = torch.cat(user_feature_emb, dim=1)
        combined_user_emb = self.userEmbNorm(combined_user_emb)
        encoding = self.userFeatureKernel(combined_user_emb).sum(dim=1)
        
        reg = torch.mean(user_id_emb * user_id_emb)
        return encoding, reg
        
    def get_item_encoding(self, item_ids, item_features, B):
        # 物品ID嵌入
        item_id_emb = self.iIDEmb(item_ids).view(B, -1, self.item_latent_dim)
        L = item_id_emb.shape[1]
        
        # 收集所有物品特征嵌入
        item_feature_emb = [item_id_emb]
        for f, fEmbModule in self.iFeatureEmb.items():
            f_dim = self.item_feature_dims[f]
            feature_emb = fEmbModule(item_features[f].view(B, L, f_dim))
            item_feature_emb.append(feature_emb.view(B, -1, self.item_latent_dim))

        # 合并并编码
        combined_item_emb = torch.cat(item_feature_emb, dim=-1).view(B, L, -1, self.item_latent_dim)
        combined_item_emb = self.itemEmbNorm(combined_item_emb)
        encoding = self.itemFeatureKernel(combined_item_emb).sum(dim=2)
        encoding = encoding.view(B, -1, self.enc_dim)
        encoding = self.encNorm(encoding)
        
        reg = torch.mean(item_id_emb * item_id_emb)
        return encoding, reg
        
    def get_response_embedding(self, feed_dict, B):
        # 收集所有历史反馈行为
        resp_list = []
        for f in self.feedback_types:
            resp = feed_dict[f'history_{f}'].view(B, self.max_len)
            resp_list.append(resp)
        
        # 编码为向量
        combined_resp = torch.cat(resp_list, dim=-1).view(B, self.max_len, self.feedback_dim)
        resp_emb = self.feedbackEncoder(combined_resp)
        return resp_emb
    
    def get_loss(self, feed_dict: dict, out_dict: dict):
        B = feed_dict['user_id'].shape[0]
        preds = out_dict['preds'].view(B, -1, self.feedback_dim)
        targets = {f: feed_dict[f].view(B, -1).to(torch.float) for f in self.feedback_types}
        loss_weight = feed_dict['loss_weight'].view(B, -1, self.feedback_dim)
        
        if self.loss_type == 'bce':
            behavior_loss = {}
            loss = 0
            for i, fb in enumerate(self.feedback_types):
                if self.behavior_weight[i] == 0:
                    continue
                Y = targets[fb].view(-1)
                P = preds[:, :, i].view(-1)
                W = loss_weight[:, :, i].view(-1)
                
                # 计算加权BCE损失
                point_loss = self.bce_loss(self.sigmoid(P), Y)
                behavior_loss[fb] = torch.mean(point_loss).item()
                point_loss = torch.mean(point_loss * W)
                loss = self.behavior_weight[i] * point_loss + loss
        else:
            raise NotImplemented
            
        out_dict['loss'] = loss + self.l2_coef * out_dict['reg']
        out_dict['behavior_loss'] = behavior_loss
        return out_dict
    
    def set_behavior_hyper_weight(self, weight):
        # 设置不同行为的损失权重
        self.behavior_weight = weight.view(-1)
        assert len(self.behavior_weight) == self.feedback_dim
