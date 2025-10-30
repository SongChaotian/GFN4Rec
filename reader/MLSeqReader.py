import numpy as np
import pandas as pd
from tqdm import tqdm

from reader.BaseReader import BaseReader
from utils import padding_and_clip, get_onehot_vocab, get_multihot_vocab


class MLSeqReader(BaseReader):
    """
    MovieLens序列数据读取器
    按单个交互记录组织数据，用于用户模拟器训练
    """
    
    @staticmethod
    def parse_data_args(parser):
        parser = BaseReader.parse_data_args(parser)
        parser.add_argument('--user_meta_file', type=str, required=True, help='user raw feature file_path')
        parser.add_argument('--item_meta_file', type=str, required=True, help='item raw feature file_path')
        parser.add_argument('--max_hist_seq_len', type=int, default=100, help='maximum history length in the input sequence')
        parser.add_argument('--val_holdout_per_user', type=int, default=5, help='number of holdout records for val set')
        parser.add_argument('--test_holdout_per_user', type=int, default=5, help='number of holdout records for test set')
        parser.add_argument('--meta_file_sep', type=str, default=',', help='separater of user/item meta csv file')
        return parser
        
    def log(self):
        super().log()
        
    def __init__(self, args):
        print("initiate MovieLens sequence reader")
        self.max_hist_seq_len = args.max_hist_seq_len
        self.val_holdout_per_user = args.val_holdout_per_user
        self.test_holdout_per_user = args.test_holdout_per_user
        super().__init__(args)
        
    def _read_data(self, args):
        """
        读取并处理数据
        构建：用户/物品词表、特征词表、用户历史、数据划分
        """
        print(f"Loading data files")

        # 1. 读取交互日志
        self.log_data = pd.read_table(args.train_file, sep=args.data_separator)
        
        # 2. 构建ID词表（从1开始编码，0留给padding）
        self.users = list(self.log_data['user_id'].unique())
        self.user_id_vocab = {uid: i+1 for i, uid in enumerate(self.users)}

        self.items = list(self.log_data['movie_id'].unique())
        self.item_id_vocab = {iid: i+1 for i, iid in enumerate(self.items)}
        
        # 3. 构建用户历史索引（每个用户的所有交互行号）
        self.user_history = {
            uid: list(self.log_data[self.log_data['user_id'] == uid].index)
            for uid in self.users
        }
        
        # 4. 加载元数据
        print("Load item meta data")
        item_meta_file = pd.read_csv(args.item_meta_file, sep=args.meta_file_sep)
        self.item_meta = item_meta_file.set_index('movie_id').to_dict('index')
        # 格式: {movie_id: {'genres': 'Action|Adventure'}}

        print("Load user meta data")
        user_meta_file = pd.read_csv(args.user_meta_file, sep=args.meta_file_sep)
        self.user_meta = user_meta_file.set_index('user_id').to_dict('index')
        # 格式: {user_id: {'gender': 'M', 'age': 25}}
        
        # 5. 构建特征词表（one-hot/multi-hot编码）
        self.selected_item_features = ['genres']
        self.selected_user_features = ['gender', 'age']
        
        self.user_vocab = get_onehot_vocab(user_meta_file, self.selected_user_features)
        self.item_vocab = {}
        self.item_vocab.update(get_multihot_vocab(item_meta_file, ['genres']))
        self.padding_item_meta = {
            f: np.zeros_like(list(v_dict.values())[0]) 
            for f, v_dict in self.item_vocab.items()
        }
        
        # 6. 定义反馈类型
        self.response_list = ['is_click', 'is_like', 'is_star']
        self.response_dim = len(self.response_list)
        self.padding_response = {resp: 0. for resp in self.response_list}
        self.response_neg_sample_rate = self.get_response_weights()
        
        # 7. 划分数据集
        # {'train': [row_id], 'val': [row_id], 'test': [row_id]}
        self.data = self._sequence_holdout(args)
        
    def _sequence_holdout(self, args):
        """
        按用户时间序列划分训练/验证/测试集
        - 每个用户保留最后 test_holdout_per_user 条作为测试集
        - 倒数第 val_holdout_per_user 条作为验证集
        - 其余作为训练集
        - 过滤训练集占比<60%的用户
        """
        print(f"sequence holdout for users (-1, {args.val_holdout_per_user}, {args.test_holdout_per_user})")
        
        if args.val_holdout_per_user == 0 and args.test_holdout_per_user == 0:
            return {"train": self.log_data.index, "val": [], "test": []}

        data = {"train": [], "val": [], "test": []}

        for u in tqdm(self.users):
            sub_df = self.log_data[self.log_data['user_id'] == u]
            n_train = len(sub_df) - args.val_holdout_per_user - args.test_holdout_per_user

            # 过滤训练数据过少的用户（训练集<60%）
            if n_train < 0.6 * len(sub_df):
                continue

            # 按时间顺序划分
            data['train'].append(list(sub_df.index[:n_train]))
            data['val'].append(list(sub_df.index[n_train:n_train+args.val_holdout_per_user]))
            data['test'].append(list(sub_df.index[-args.test_holdout_per_user:]))

        # 合并所有用户的索引
        for k, v in data.items():
            data[k] = np.concatenate(v)

        return data
    
    def get_response_weights(self):
        """计算负样本权重（正样本数/负样本数）"""
        ratio = {}
        for f in self.response_list:
            counts = self.log_data[f].value_counts()
            ratio[f] = float(counts[1]) / counts[0]
        return ratio

    ###########################
    #        Iterator         #
    ###########################
        
    def __getitem__(self, idx):
        """
        获取单个交互样本
        
        Returns:
            record: {
                'user_id': 编码后的用户ID,
                'item_id': 编码后的物品ID,
                'is_click/is_like/is_star': 反馈标签,
                'uf_{feature}': 用户特征,
                'if_{feature}': 物品特征,
                'history': (max_H,) 历史物品ID,
                'history_length': 历史长度,
                'history_if_{feature}': (max_H, feature_dim) 历史物品特征,
                'history_{response}': (max_H,) 历史反馈,
                'loss_weight': (n_response,) 损失权重
            }
        """
        # 1. 获取当前交互记录
        row_id = self.data[self.phase][idx]
        row = self.log_data.iloc[row_id]
        
        user_id = row['user_id']
        item_id = row['movie_id']
    
        record = {
            'user_id': self.user_id_vocab[user_id],
            'item_id': self.item_id_vocab[item_id],
            'is_click': row['is_click'],
            'is_like': row['is_like'],
            'is_star': row['is_star']
        }
        
        # 2. 添加用户特征
        user_meta = self.get_user_meta_data(user_id)
        record.update(user_meta)

        # 3. 添加物品特征
        item_meta = self.get_item_meta_data(item_id)
        record.update(item_meta)
        
        # 4. 构建用户历史（当前交互之前的最多max_hist_seq_len条）
        H_rowIDs = [rid for rid in self.user_history[user_id] if rid < row_id][-self.max_hist_seq_len:]
        history, hist_length, hist_meta, hist_response = self.get_user_history(H_rowIDs)

        record['history'] = np.array(history)  # (50,) 编码后的item_id序列
        record['history_length'] = hist_length

        # 历史物品特征
        for f, v in hist_meta.items():
            record[f'history_{f}'] = v
        
        # 历史反馈
        for f, v in hist_response.items():
            record[f'history_{f}'] = v
            
        # 5. 计算损失权重（负样本降权）
        loss_weight = np.array([
            1. if record[f] == 1 else self.response_neg_sample_rate[f] 
            for f in self.response_list
        ])
        record["loss_weight"] = loss_weight

        return record
    
    def get_user_meta_data(self, user_id):
        """提取用户特征（one-hot编码）"""
        user_feature_dict = self.user_meta[user_id]
        user_meta_record = {
            f'uf_{f}': self.user_vocab[f][user_feature_dict[f]]
            for f in self.selected_user_features
        }
        return user_meta_record
    
    def get_item_meta_data(self, item_id):
        """提取物品特征（multi-hot编码）"""
        item_feature_dict = self.item_meta[item_id]
        item_meta_record = {}
        # genres是逗号分隔的多值，需要累加multi-hot向量
        item_meta_record['if_genres'] = np.sum(
            [self.item_vocab['genres'][g] for g in item_feature_dict['genres'].split(',')], 
            axis=0
        )
        return item_meta_record
    
    def get_user_history(self, H_rowIDs):
        """
        构建用户历史序列
        
        Returns:
            history: (max_H,) 物品ID序列（padding+真实历史）
            hist_length: 实际历史长度
            hist_meta: {if_{feature}: (max_H, feature_dim)} 历史物品特征
            history_response: {response_type: (max_H,)} 历史反馈
        """
        L = len(H_rowIDs)

        if L == 0:
            # 空历史：全部填充padding
            history = [0] * self.max_hist_seq_len
            hist_meta = {
                f'if_{f}': np.tile(self.padding_item_meta[f], self.max_hist_seq_len)
                for f in self.selected_item_features
            }
            history_response = {
                resp: np.array([self.padding_response[resp]] * self.max_hist_seq_len)
                for resp in self.response_list
            }
        else:
            H = self.log_data.iloc[H_rowIDs]
            
            # 1. 历史物品ID（左padding）
            item_ids = [self.item_id_vocab[iid] for iid in H['movie_id']] 
            history = padding_and_clip(item_ids, self.max_hist_seq_len)

            # 2. 历史物品特征（左padding）
            meta_list = [self.get_item_meta_data(iid) for iid in H['movie_id']] 
            hist_meta = {} 
            for f in self.selected_item_features:
                padding = [self.padding_item_meta[f] for _ in range(self.max_hist_seq_len - L)]
                real_hist = [v_dict[f'if_{f}'] for v_dict in meta_list]
                hist_meta[f'if_{f}'] = np.concatenate(padding + real_hist, axis=0)

            # 3. 历史反馈（左padding）
            history_response = {}
            for resp in self.response_list:
                padding = np.array([self.padding_response[resp]] * (self.max_hist_seq_len - L))
                real_resp = np.array(H[resp])
                history_response[resp] = np.concatenate([padding, real_resp], axis=0)
                
        return history, L, hist_meta, history_response

    def get_statistics(self):
        """返回数据集统计信息"""
        stats = {}
        stats["raw_data_size"] = len(self.log_data)
        stats["data_size"] = [len(self.data['train']), len(self.data['val']), len(self.data['test'])]
        stats["n_user"] = len(self.users)
        stats["n_item"] = len(self.items)
        stats["max_seq_len"] = self.max_hist_seq_len
        stats["user_features"] = self.selected_user_features
        stats["user_feature_dims"] = {f: len(list(v_dict.values())[0]) for f, v_dict in self.user_vocab.items()}
        stats["item_features"] = self.selected_item_features
        stats["item_feature_dims"] = {f: len(list(v_dict.values())[0]) for f, v_dict in self.item_vocab.items()}
        stats["feedback_type"] = self.response_list
        stats["feedback_size"] = self.response_dim
        stats["feedback_negative_sample_rate"] = self.response_neg_sample_rate
        return stats