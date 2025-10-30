# ML1M数据格式和划分详解

---

## 1. 用户模拟器训练的数据格式

### 1.1 训练配置

**脚本**: `train_multi_behavior_user_response_movielens.sh`

```bash
python train_multibehavior.py\
    --reader MLSeqReader\                    # 数据读取器
    --train_file ../dataset/ml1m/log_session.csv\
    --max_hist_seq_len 50\
    --val_holdout_per_user 5\                # 验证集: 每用户5个交互
    --test_holdout_per_user 5\               # 测试集: 每用户5个交互
    --model KRMBUserResponse_MaxOut\
    --batch_size 128
```

### 1.2 数据划分方式

**来自**: `reader/MLSeqReader.py`

```python
def _sequence_holdout(self, args):
    """
    按用户时间序列划分训练/验证/测试集
    """
    data = {"train": [], "val": [], "test": []}

    for u in tqdm(self.users):
        sub_df = self.log_data[self.log_data['user_id'] == u]
        n_train = len(sub_df) - args.val_holdout_per_user - args.test_holdout_per_user

        # 过滤掉训练数据过少的用户（训练集<60%总数据）
        if n_train < 0.6 * len(sub_df):
            continue

        # 按时间顺序划分
        data['train'].append(list(sub_df.index[:n_train]))
        data['val'].append(list(sub_df.index[n_train:n_train+args.val_holdout_per_user]))
        data['test'].append(list(sub_df.index[-args.test_holdout_per_user:]))

    # 合并所有用户的数据
    for k, v in data.items():
        data[k] = np.concatenate(v)

    return data
```

**划分示例**（假设用户有100条交互）:
```
用户1的交互序列（按时间排序）:
├─ 训练集: index [0:90]    → 90条交互（前90个）
├─ 验证集: index [90:95]   → 5条交互（第91-95个）
└─ 测试集: index [95:100]  → 5条交互（最后5个）

合并后的数据格式：
data = {
    'train': [所有用户的训练集行索引拼接],
    'val': [所有用户的验证集行索引拼接],
    'test': [所有用户的测试集行索引拼接]
}
```

**关键点**:
1. `val_holdout_per_user=5` 和 `test_holdout_per_user=5` 指的是**交互记录数量**，不是slate数量
2. **用户过滤**：只保留训练集占比≥60%的用户（默认参数下至少需要25条交互）
3. 数据按**单个交互记录**组织，不涉及slate概念
4. 返回的是行索引（row index），用于从DataFrame中提取数据

### 1.3 批次数据格式

**来自**: `reader/MLSeqReader.py`

```python
def __getitem__(self, idx):
    '''
    返回单个样本:
    {
        'user_id': (1,)           # 标量
        'item_id': (1,)           # 标量
        'is_click': (1,)          # 标量
        'is_like': (1,)           # 标量
        'is_star': (1,)           # 标量
        'uf_{feature}': (F_dim,)  # 用户特征向量
        'if_{feature}': (F_dim,)  # 物品特征向量
        'history': (max_H,)       # 历史物品ID
        'history_length': (1,)    # 标量
        'history_if_{feature}': (max_H, F_dim)
        'history_{response}': (max_H,)
    }
    '''
```

**DataLoader collate后的batch**:
```python
{
    'user_id': (B,)           # 批次中的用户ID
    'item_id': (B,)           # 批次中的物品ID（单个）
    'is_click': (B,)          # 点击标签
    'is_like': (B,)           # 点赞标签
    'is_star': (B,)           # 收藏标签
    'uf_gender': (B, 2),      # 用户性别特征
    'uf_age': (B, 7),         # 用户年龄特征
    'if_genres': (B, 18),     # 物品类型特征
    'history': (B, 50),       # 用户历史
    'history_length': (B,),
    'history_if_genres': (B, 50, 18),
    'history_is_click': (B, 50),
    'history_is_like': (B, 50),
    'history_is_star': (B, 50),
}
```

**重要**: 用户模拟器训练时，每个样本是**单个交互记录**（一个用户对一个物品的反馈），不是一个slate。

---

## 2. 离线训练GFN的数据格式

### 2.1 训练配置

**脚本**: `train_offline_gfn_db_movielens.sh`

```bash
python train_online_policy.py\
    --env_class MLUserEnvironment_ListRec\
    --slate_size 6\                          # ⭐ slate长度为6
    --new_reader_class MLSlateReader\        # ⭐ 使用slate读取器
    --env_test_holdout 1\                    # ⭐ 测试集holdout=1
    --policy_class SlateGFN_DB\
    --agent_class OfflineAgentWithOnlineTest\# ⭐ 离线训练Agent
    --n_iter 10000\                          # 10000次迭代
    --batch_size 128
```

### 2.2 关键差异

| 维度 | 用户模拟器训练 | 离线GFN训练 |
|------|----------------|-------------|
| **数据读取器** | `MLSeqReader` | `MLSlateReader` |
| **slate_size** | 未指定 | 6 |
| **测试集** | 每用户5个交互 | 每用户1个slate（6个交互）|
| **Agent** | `KRMBUserResponse_MaxOut` | `OfflineAgentWithOnlineTest` |

### 2.3 数据划分方式

**来自**: `reader/MLSlateReader.py`

```python
class MLSlateReader(MLSeqReader):
    def __init__(self, args):
        self.slate_size = args.slate_size  # slate_size = 6
        super().__init__(args)
        
    def _sequence_holdout(self, args):
        data = {"train": [], "val": [], "test": []}
        
        for u in tqdm(self.users):
            sub_df = self.log_data[self.log_data['user_id'] == u]
            
            # ⭐ 核心：计算训练集大小（考虑slate_size）
            n_train = len(sub_df) - (args.val_holdout_per_user + args.test_holdout_per_user) * self.slate_size
            
            # ⭐ 训练集：每隔slate_size取一个起始索引
            data['train'].append(list(sub_df.index[:n_train])[::self.slate_size])
            
            # 验证集：连续的 val_holdout_per_user * slate_size 个交互
            data['val'].append(list(sub_df.index[n_train:n_train+args.val_holdout_per_user*self.slate_size]))
            
            # ⭐ 测试集：最后 test_holdout_per_user * slate_size 个交互，每隔slate_size取一个起始索引
            data['test'].append(list(sub_df.index[-args.test_holdout_per_user*self.slate_size::self.slate_size]))
        
        for k,v in data.items():
            data[k] = np.concatenate(v).astype(int)
        
        return data
```

**参数值**:
```bash
slate_size = 6
val_holdout_per_user = 5   # 从父类MLSeqReader继承，默认为5
test_holdout_per_user = 1  # 通过 --env_test_holdout 传递
```

**计算过程**:
```python
# 假设用户有100条交互
n_train = 100 - (5 + 1) * 6 = 64

# 训练集：前64个交互，每隔6个取一个起始索引
train_indices = [0, 6, 12, 18, 24, ..., 60]  # 约11个slate起始点

# 验证集：接下来30个交互（连续索引）
val_indices = [64, 65, 66, ..., 93]  # 30个连续索引（5个slate）

# 测试集：最后6个交互，每隔6个取一个起始索引
test_indices = [94]  # 1个slate起始点
```

**划分示例**（假设用户有100条交互）:
```
用户1的交互序列:
├─ 训练集索引: [0, 6, 12, 18, ..., 60]  → 11个slate起始点
├─ 验证集索引: [64, 65, 66, ..., 93]    → 30个连续索引（5个slate）
└─ 测试集索引: [94]                     → 1个slate起始点

每个索引代表一个slate的起始位置:
- slate_0: 交互[0:6]
- slate_1: 交互[6:12]
- ...
- slate_10: 交互[60:66]
- val_slate_0: 交互[64:70]
- val_slate_1: 交互[70:76]
- ...
- val_slate_4: 交互[88:94]
- test_slate: 交互[94:100]
```

**关键理解**:
1. `[::self.slate_size]` - Python切片语法，表示从0开始，每隔`slate_size`取一个元素
2. 训练集和测试集只存储**slate的起始索引**，而不是所有交互的索引
3. 验证集存储**连续的索引**（用于构建完整的slate序列）
4. **重要修正**：`val_holdout_per_user`参数从父类继承，默认值为5（不是0）

### 2.4 批次数据格式

**来自**: `reader/MLSlateReader.py`

```python
def __getitem__(self, idx):
    '''
    返回单个slate样本:
    {
        'user_id': (1,)
        'item_id': (slate_size,)           # slate中的物品序列
        'is_click': (slate_size,)          # slate中每个物品的点击
        'is_like': (slate_size,)
        'is_star': (slate_size,)
        'uf_{feature}': (F_dim,)
        'if_{feature}': (slate_size, F_dim)# slate中每个物品的特征
        'history': (max_H,)
        'history_length': (1,)
        'history_if_{feature}': (max_H, F_dim)
        'history_{response}': (max_H,)
    }
    '''
```

**DataLoader collate后的batch**:
```python
{
    'user_id': (B,)              # B个用户
    'item_id': (B, 6),           # B个slate，每个包含6个物品
    'is_click': (B, 6),          # B个slate的点击标签
    'is_like': (B, 6),
    'is_star': (B, 6),
    'uf_gender': (B, 2),
    'uf_age': (B, 7),
    'if_genres': (B, 6, 18),     # B个slate中每个物品的类型特征
    'history': (B, 50),
    'history_length': (B,),
    'history_if_genres': (B, 50, 18),
    'history_is_click': (B, 50),
    'history_is_like': (B, 50),
    'history_is_star': (B, 50),
}
```

**重要**: 离线GFN训练时，每个样本是**一个slate**（一个用户与连续6个物品的交互序列）。

### 2.5 训练数据使用

**离线训练时使用DataLoader**:
```python
# 来自 OfflineAgentWithOnlineTest
def action_before_train(self):
    """初始化DataLoader"""
    self.offline_iter = iter(DataLoader(
        self.env.reader,          # MLSlateReader
        batch_size=128,
        shuffle=True
    ))

def step_train(self):
    """从DataLoader采样并训练"""
    batch_sample = next(self.offline_iter)
    
    # 提取监督标签
    target_action = batch_sample['item_id'] - 1      # (B, 6)
    target_response = batch_sample[响应类型]         # (B, 6, 3)
    
    # GFN前向传播（teacher forcing）
    policy_output = self.actor(
        observation, 
        action=target_action,      # 使用离线数据的动作
        response=target_response   # 使用离线数据的反馈
    )
    
    # 计算GFN损失（DB损失）
    loss = self.actor.get_loss(input_dict, policy_output)
    loss.backward()
```

---

## 3. 在线训练GFN的数据格式

### 3.1 训练配置

**脚本**: `train_gfn_db_movielens.sh`

```bash
python train_online_policy.py\
    --env_class MLUserEnvironment_ListRec\
    --slate_size 6\
    --agent_class BaseOnlineAgent\          # ⭐ 在线训练Agent
    --n_iter 5000\                          # 5000次迭代
    --episode_batch_size 128\               # ⭐ 每次交互128个用户
    --batch_size 128\                       # ⭐ Buffer采样128个样本
    --buffer_class SequentialBuffer\        # ⭐ 使用Buffer
    --start_train_at_step 100\              # ⭐ 前100步随机探索
    --initial_greedy_epsilon 0.05\          # ⭐ 探索率0.05→0.01
    --final_greedy_epsilon 0.01
```

### 3.2 关键差异

| 维度 | 离线GFN训练 | 在线GFN训练 |
|------|-------------|-------------|
| **Agent** | `OfflineAgentWithOnlineTest` | `BaseOnlineAgent` |
| **数据来源** | DataLoader（离线日志）| 环境实时交互 |
| **数据读取器** | `MLSlateReader` | 不需要 |
| **训练步数** | 10000 steps | 5000 steps |
| **Buffer** | 不使用 | `SequentialBuffer`（50000容量）|
| **探索策略** | 无 | ε-greedy（0.05→0.01）|

### 3.3 数据来源

**在线训练不使用静态数据集**，数据来源于：

1. **环境采样**: 每次从用户集合中随机采样128个用户
2. **策略交互**: 策略为每个用户生成推荐列表（slate）
3. **模拟器反馈**: 用户模拟器预测用户对slate的反馈
4. **Buffer存储**: 将交互经验（s, a, r, s'）存入Buffer
5. **Buffer采样**: 训练时从Buffer中采样批次数据

### 3.4 数据流

**来自**: `model/agent/BaseOnlineAgent.py`

```python
def action_before_train(self):
    """训练前准备：随机探索填充Buffer"""
    self.buffer.reset(self.env, self.actor)
    observation = self.env.reset()  # 采样128个用户
    
    for i in range(100):  # start_train_at_step=100
        observation = self.run_episode_step(
            0, 
            epsilon=1.0,  # 完全随机探索
            observation, 
            do_buffer_update=True,  # 存入Buffer
            do_explore=True
        )
    
    return observation

def train(self):
    """在线训练主循环"""
    observation = self.action_before_train()
    
    for i in range(5000):  # N_ITER=5000
        epsilon = self.exploration_scheduler.value(i)  # 0.05→0.01
        
        # 【核心1】与环境交互产生新数据
        observation = self.run_episode_step(
            i, epsilon, observation, 
            do_buffer_update=True,  # 存入Buffer
            do_explore=True         # 启用探索
        )
        
        # 【核心2】从Buffer采样训练
        if i % 1 == 0:  # train_every_n_step=1
            self.step_train()

def run_episode_step(self, epsilon, observation, do_buffer_update, do_explore):
    """在线交互一步"""
    # 1. 策略生成动作（带ε-greedy探索）
    policy_output = self.actor({
        'observation': observation,
        'epsilon': epsilon,
        'do_explore': do_explore
    })
    # policy_output['action']: (128, 6)
    
    # 2. 环境执行动作，获取反馈
    new_observation, user_feedback, updated_observation = \
        self.env.step({'action': policy_output['action']})
    # user_feedback['immediate_response']: (128, 6, 3)
    
    # 3. 计算奖励
    R = self.reward_func(user_feedback)  # (128,)
    
    # 4. 存入Buffer
    if do_buffer_update:
        self.buffer.update(
            observation,         # s_t
            policy_output,       # a_t, π(a|s)
            user_feedback,       # r_t
            updated_observation  # s_{t+1}
        )
    
    return new_observation

def step_train(self):
    """从Buffer采样并训练"""
    # 从Buffer采样128个经验
    observation, target_output, target_response, _, __ = \
        self.buffer.sample(128)
    
    # 前向传播
    policy_output = self.actor({
        'observation': observation,
        'action': target_output['action'],  # Buffer中的动作
        'response': target_response,        # Buffer中的反馈
        'is_train': True
    })
    
    # 计算GFN损失
    loss = self.actor.get_loss(input_dict, policy_output)
    loss.backward()
```

### 3.5 环境数据格式

**环境重置** (`env.reset()`):
```python
observation = {
    'user_id': (128,),              # 128个随机采样的用户
    'uf_gender': (128, 2),
    'uf_age': (128, 7),
    'history': (128, 50),
    'history_length': (128,),
    'history_if_genres': (128, 50, 18),
    'history_is_click': (128, 50),
    'history_is_like': (128, 50),
    'history_is_star': (128, 50),
}
```

**环境交互** (`env.step(action)`):
```python
# 输入
action = (128, 6)  # 128个用户，每人6个推荐物品

# 输出
user_feedback = {
    'immediate_response': (128, 6, 3),  # 3种反馈：click, like, star
    'reward': (128,),                   # 每个用户的总奖励
    'coverage': scalar,                 # 覆盖率
    'ILD': scalar,                      # 列表内多样性
}

new_observation = {...}  # 更新后的用户状态（暂不更新，返回原状态）
```

### 3.6 Buffer结构

**Buffer存储的经验**:
```python
experience = {
    'observation': {
        'user_id': (128,),
        ...
    },
    'policy_output': {
        'action': (128, 6),      # 选择的物品
        'prob': (128, 6),        # 选择概率
        'logF': (128, 7),        # DB: 流值
    },
    'user_feedback': {
        'immediate_response': (128, 6, 3),
        'reward': (128,),
    },
    'next_observation': {...}
}

# Buffer容量: 50000条经验
# 采样: 每次采样128个经验用于训练
```

---

## 4. 三种训练方式的完整对比

### 4.1 配置对比表

| 维度 | 用户模拟器训练 | 离线GFN训练 | 在线GFN训练 |
|------|----------------|-------------|-------------|
| **数据读取器** | `MLSeqReader` | `MLSlateReader` | 不需要 |
| **数据来源** | 离线日志（按交互）| 离线日志（按slate）| 环境实时交互 |
| **slate_size** | 未指定 | 6 | 6 |
| **测试集** | 每用户5个交互 | 每用户1个slate（6个交互）| 最后100步 |
| **训练方式** | 监督学习（BCE）| 监督学习（GFN）| 强化学习（GFN）|
| **训练迭代** | 10 epochs | 10000 steps | 5000 steps |
| **批次大小** | 128 | 128 | 128（Buffer采样）|
| **episode_batch_size** | N/A | 128（仅评估用）| 128（交互用户数）|
| **探索策略** | 无 | 无 | ε-greedy（0.05→0.01）|
| **Buffer** | 不使用 | 不使用 | 50000容量 |
| **冷启动** | N/A | N/A | 100步随机探索 |

### 4.2 数据格式对比

#### 用户模拟器训练的batch:
```python
{
    'user_id': (B,),
    'item_id': (B,),          # 单个物品
    'is_click': (B,),         # 单个标签
    'is_like': (B,),
    'is_star': (B,),
    ...
}
```

#### 离线GFN训练的batch:
```python
{
    'user_id': (B,),
    'item_id': (B, 6),        # slate（6个物品）
    'is_click': (B, 6),       # slate的标签
    'is_like': (B, 6),
    'is_star': (B, 6),
    ...
}
```

#### 在线GFN训练的batch（从Buffer采样）:
```python
{
    'observation': {
        'user_id': (B,),
        ...
    },
    'action': (B, 6),         # 策略生成的slate
    'immediate_response': (B, 6, 3),  # 模拟器预测的反馈
    'reward': (B,),
    ...
}
```

---

## 5. 关键代码位置索引

### 5.1 数据读取器
- `reader/MLSeqReader.py` - 用户模拟器训练的数据读取
  - `_sequence_holdout()` - 数据划分逻辑（按交互）
  - `__getitem__()` - 返回单个交互样本
  
- `reader/MLSlateReader.py` - 离线GFN训练的数据读取
  - `_sequence_holdout()` - 数据划分逻辑（按slate）
  - `__getitem__()` - 返回单个slate样本

### 5.2 训练脚本
- `train_multibehavior.py` - 用户模拟器训练主程序
- `train_online_policy.py` - GFN训练主程序（离线和在线共用）

### 5.3 Agent
- `model/agent/OfflineAgentWithOnlineTest.py` - 离线GFN训练
  - `action_before_train()` - 初始化DataLoader
  - `step_train()` - 从DataLoader训练
  
- `model/agent/BaseOnlineAgent.py` - 在线GFN训练
  - `action_before_train()` - 随机探索填充Buffer
  - `run_episode_step()` - 在线交互
  - `step_train()` - 从Buffer训练

---

## 6. 论文与代码的对应

### 6.1 数据划分

**论文描述**:
> "To engage in offline test, we split the last N interactions of each user's history as test samples while the remaining as training samples, and we set N = 1 for ML1M"

**代码实现** (`reader/MLSlateReader.py`):
```python
# N = 1 指的是 1个slate
test_holdout_per_user = 1
slate_size = 6

# 因此测试集是最后 1 * 6 = 6 个交互
n_train = len(sub_df) - (5 + 1) * 6  # 注意：val_holdout_per_user=5

# 测试集索引（每隔6个取一个，只取1个）
test_indices = list(sub_df.index[-6::6])  # 只有1个索引
```

**结论**: N=1 在代码中的含义是**1个slate**（6个连续交互），而不是1个交互记录。

---

## 总结

1. **用户模拟器训练**：
   - 使用`MLSeqReader`按**单个交互记录**组织数据
   - 每用户保留最后5个交互作为验证集和测试集
   - 训练目标：预测单个物品的用户反馈

2. **离线GFN训练**：
   - 使用`MLSlateReader`按**slate**组织数据（slate_size=6）
   - 每用户保留最后1个slate（6个交互）作为测试集，5个slate（30个交互）作为验证集
   - 使用DataLoader从离线日志中采样
   - 训练目标：学习生成高质量推荐列表

3. **在线GFN训练**：
   - 不使用数据读取器，通过环境实时交互产生数据
   - 每次交互采样128个用户，生成128个slate
   - 使用Buffer存储经验（容量50000）
   - 采用ε-greedy策略探索（0.05→0.01）
   - 训练目标：通过探索学习最优推荐策略

**核心区别**：
- 用户模拟器：单个交互 → 预测反馈
- 离线GFN：离线slate → 监督学习策略
- 在线GFN：在线slate → 强化学习策略