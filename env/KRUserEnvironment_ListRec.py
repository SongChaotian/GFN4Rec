import numpy as np
import utils
import torch
import random
from copy import deepcopy
from argparse import Namespace
from torch.utils.data import DataLoader
from torch.distributions import Categorical

from reader import *
from env.KRUserEnvironment_FiniteImmediate import KRUserEnvironment_FiniteImmediate


class KRUserEnvironment_ListRec(KRUserEnvironment_FiniteImmediate):
    '''
    KuaiRand列表推荐环境
    继承自FiniteImmediate环境，专门用于列表推荐场景
    '''
    
    @staticmethod
    def parse_model_args(parser):
        '''继承父类参数配置'''
        parser = KRUserEnvironment_FiniteImmediate.parse_model_args(parser)
        return parser
    
    def __init__(self, args):
        '''初始化环境，继承父类所有属性'''
        super().__init__(args)
        
    def reset(self):
        '''
        重置环境，采样新用户批次
        @output: observation字典，包含用户画像和历史
        '''
        BS = self.episode_batch_size
        self.iter = iter(DataLoader(
            self.reader, 
            batch_size=BS, 
            shuffle=True, 
            pin_memory=True, 
            num_workers=8
        ))
        
        initial_sample = next(self.iter)
        self.current_observation = self.get_observation_from_batch(initial_sample)
        self.temper = torch.ones(self.episode_batch_size).to(self.device) * self.initial_temper
        self.user_step_count = 0
        
        return deepcopy(self.current_observation)
    
    def step(self, step_dict):
        '''
        执行一步交互
        @input: {'action': (B, slate_size)} 推荐动作
        @output: (新观察, 用户反馈, 更新后的观察)
        '''
        action = step_dict['action']  # (B, slate_size)
        
        with torch.no_grad():
            # 获取用户响应
            response_out = self.get_response(step_dict)
            response = response_out['immediate_response']  # (B, slate_size, n_feedback)
            
            # 判断用户是否离开
            done_mask = self.get_leave_signal(response)  # (B,)
            
            # 更新观察
            update_info = self.update_observation(action, response, done_mask)
            self.user_step_count += 1
            
            # 计算平均反馈（用于记录）
            for i, f in enumerate(self.response_types):
                R = response.mean(1)[:, i].detach()  # (B,)
            
            # 所有用户同时结束时，采样新用户
            if done_mask.sum() == len(done_mask):
                try:
                    sample_info = next(self.iter)
                    if sample_info['user_profile'].shape[0] != len(done_mask):
                        raise StopIteration
                except:
                    self.iter = iter(DataLoader(
                        self.reader, 
                        batch_size=done_mask.shape[0], 
                        shuffle=True, 
                        pin_memory=True, 
                        num_workers=8
                    ))
                    sample_info = next(self.iter)
                
                new_observation = self.get_observation_from_batch(sample_info)
                self.current_observation = new_observation
                self.temper = torch.ones(self.episode_batch_size).to(self.device) * self.initial_temper
                self.user_step_count = 0
                
            elif done_mask.sum() > 0:
                print(done_mask)
                print("User leave not synchronized")
                raise NotImplementedError
        
        user_feedback = {
            'immediate_response': response, 
            'done': done_mask, 
            'coverage': response_out['coverage'], 
            'ILD': response_out['ILD']
        }
        
        return deepcopy(self.current_observation), user_feedback, update_info['updated_observation']
    
    def get_env_report(self, window=50):
        '''获取环境统计报告'''
        report = super().get_env_report(window)
        return report