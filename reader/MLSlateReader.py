import numpy as np
import pandas as pd
from tqdm import tqdm

from reader.MLSeqReader import MLSeqReader
from utils import padding_and_clip, get_onehot_vocab, get_multihot_vocab


class MLSlateReader(MLSeqReader):
    """
    MovieLens Slate数据读取器
    将连续的交互记录组织成slate（推荐列表）格式
    """
    
    @staticmethod
    def parse_data_args(parser):
        parser = MLSeqReader.parse_data_args(parser)
        return parser
        
    def log(self):
        super().log()
        
    def __init__(self, args):
        """
        初始化Slate读取器
        Args:
            args.slate_size: slate长度（推荐列表中的物品数量）
        """
        print("initiate MLMultiBehaior Slate reader")
        self.slate_size = args.slate_size
        super().__init__(args)
        
    def _sequence_holdout(self, args):
        """
        按slate划分训练/验证/测试集
        
        划分逻辑：
            - 训练集：每隔slate_size取一个起始索引
            - 验证集：连续的val_holdout_per_user*slate_size个交互
            - 测试集：最后test_holdout_per_user*slate_size个交互，每隔slate_size取一个起始索引
        """
        print(f"sequence holdout for users (-1, {args.val_holdout_per_user}, {args.test_holdout_per_user})")
        
        if args.val_holdout_per_user == 0 and args.test_holdout_per_user == 0:
            return {"train": self.log_data.index, "val": [], "test": []}
        
        data = {"train": [], "val": [], "test": []}
        
        for u in tqdm(self.users):
            sub_df = self.log_data[self.log_data['user_id'] == u]
            
            # 计算训练集大小（扣除验证集和测试集）
            n_train = len(sub_df) - (args.val_holdout_per_user + args.test_holdout_per_user) * self.slate_size
            
            # 训练集：每隔slate_size取一个slate起始点
            data['train'].append(list(sub_df.index[:n_train])[::self.slate_size])
            
            # 验证集：连续索引（用于构建完整slate序列）
            data['val'].append(list(sub_df.index[n_train:n_train + args.val_holdout_per_user * self.slate_size]))
            
            # 测试集：每隔slate_size取一个slate起始点
            data['test'].append(list(sub_df.index[-args.test_holdout_per_user * self.slate_size::self.slate_size]))
        
        # 合并所有用户的索引
        for k, v in data.items():
            data[k] = np.concatenate(v).astype(int)
        
        return data
        
    def _read_data(self, args):
        super()._read_data(args)
    
    ###########################
    #        Iterator         #
    ###########################
        
    def __getitem__(self, idx):
        """
        获取单个slate样本
        
        Returns:
            record: {
                'user_id': 编码后的用户ID,
                'item_id': (slate_size,) slate中的物品ID序列,
                'is_click/is_like/is_star': (slate_size,) slate中每个物品的反馈,
                'uf_{feature}': 用户特征,
                'if_{feature}': (slate_size, feature_dim) slate中每个物品的特征,
                'history': (max_H,) 用户历史物品ID,
                'history_length': 历史长度,
                'history_if_{feature}': (max_H, feature_dim) 历史物品特征,
                'history_{response}': (max_H,) 历史反馈
            }
        """
        # 获取slate起始位置
        row_id = self.data[self.phase][idx]
        row = self.log_data.iloc[row_id]
        user_id = row['user_id']
        
        # 构建样本记录
        record = {'user_id': self.user_id_vocab[user_id]}
        
        # 添加用户特征
        user_meta = self.get_user_meta_data(user_id)
        record.update(user_meta)
        
        # 获取slate（从row_id开始的连续slate_size个物品）
        item_id, item_meta, item_response = self.get_slate(user_id, row_id)
        record['item_id'] = item_id
        record.update(item_meta)
        record.update(item_response)
        
        # 获取用户历史（row_id之前的交互）
        H_rowIDs = [rid for rid in self.user_history[user_id] if rid < row_id][-self.max_hist_seq_len:]
        history, hist_length, hist_meta, hist_response = self.get_user_history(H_rowIDs)
        record['history'] = np.array(history)
        record['history_length'] = hist_length
        
        for f, v in hist_meta.items():
            record[f'history_{f}'] = v
        for f, v in hist_response.items():
            record[f'history_{f}'] = v
        
        return record
    
    def get_slate(self, user_id, row_id):
        """
        获取从row_id开始的连续slate_size个物品
        
        Returns:
            slate_ids: (slate_size,) 物品ID序列
            slate_meta: {if_{feature}: (slate_size, feature_dim)} 物品特征
            slate_response: {response_type: (slate_size,)} 用户反馈
        """
        # 获取从row_id开始的slate_size个交互记录
        S_rowIDs = [rid for rid in self.user_history[user_id] if row_id <= rid][:self.slate_size]
        H = self.log_data.iloc[S_rowIDs]
        
        # 提取物品ID
        slate_ids = np.array([self.item_id_vocab[iid] for iid in H['movie_id']])
        
        # 提取物品特征
        meta_list = [self.get_item_meta_data(iid) for iid in H['movie_id']]
        slate_meta = {}
        for f in self.selected_item_features:
            slate_meta[f'if_{f}'] = np.array([v_dict[f'if_{f}'] for v_dict in meta_list])
        
        # 提取用户反馈
        slate_response = {}
        for resp in self.response_list:
            slate_response[resp] = np.array(H[resp])
        
        return slate_ids, slate_meta, slate_response

    def get_statistics(self):
        """返回数据集统计信息"""
        stats = super().get_statistics()
        stats["slate_size"] = self.slate_size
        return stats