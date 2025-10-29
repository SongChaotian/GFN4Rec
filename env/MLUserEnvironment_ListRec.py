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
from env.KRUserEnvironment_ListRec import KRUserEnvironment_ListRec


class MLUserEnvironment_ListRec(KRUserEnvironment_ListRec):
    '''
    MovieLens列表推荐环境
    继承自KRUserEnvironment_ListRec，针对MovieLens数据集的特定配置
    '''
    
    @staticmethod
    def parse_model_args(parser):
        '''继承父类参数配置'''
        parser = KRUserEnvironment_ListRec.parse_model_args(parser)
        return parser
    
    def __init__(self, args):
        '''
        初始化环境
        覆盖response_weights：MovieLens所有反馈类型权重均为1
        '''
        super().__init__(args)
        # MovieLens数据集中所有反馈类型权重相同
        self.response_weights = [1 for f in self.response_types]