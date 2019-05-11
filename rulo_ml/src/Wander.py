#!/usr/bin/env	python
import os
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from collections import deque, OrderedDict
from rulo_utils.csvreader import csvread
from rulo_utils.csvwriter import csvwriter
from rulo_utils.numpywriter import numpywriter
from rulo_utils.numpyreader import numpyreader

class Wander:
    
    def __init__(self):
        if not os.path.isfile('/home/mtb/rulo_ws/src/rulo_pkg/rulo_ml/data/rewards.csv'):
            self.reward = pd.read_csv('/home/mtb/rulo_ws/src/rulo_pkg/rulo_ml/data/cells_time.csv', index_col=[0])            
            self.dirt_lev= np.load('/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/data/dirt.npy')
            for i in range(623):
                for j in range(623):
                    if i == j:
                        self.reward[str(i)][j] = self.dirt_lev[i] / 10.0
                    else:
                        self.reward[str(i)][j] = self.dirt_lev[j] / self.reward[str(i)][j]
            self.reward = self.reward.drop([10000])
            max_ =  np.amax(self.reward.max().values)
            self.reward = self.reward  / max_        
            self.reward.to_csv('/home/mtb/rulo_ws/src/rulo_pkg/rulo_ml/data/rewards.csv')
        else:
            self.reward = pd.read_csv('/home/mtb/rulo_ws/src/rulo_pkg/rulo_ml/data/rewards.csv', index_col=[0])
        
        self.state = np.load('/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/data/pose.npy')                    

        # np.random.seed(12356)
    def initialize(self):
        num = 24
        return self.state[num].reshape([1,2])

    def reset(self):
        # self.num = 25
        # return self.state[self.num].reshape([1, 2])
        index =  np.random.randint(0, 50)
        return self.state[index].reshape([1,2])
    

    def step(self,s,a):
        action = np.argmax(a)
        next_state = np.reshape(self.state[action],(1,2))
        if action in self.cleaned_cells:
            if len(self.cleaned_cells) == 50:
                terminal =True
                update = True
                next_state = self.state[24].reshape([1, 2])
                action = 24                         
            else:
                terminal = False    
                update = False 
                next_state = s 
            reward = 0.0                                     
        else:
            reward=self.get_reward(s,next_state)
            self.all_reward +=reward              
            self.cleaned_cells.append(action)
            update = True 
            terminal = False      
                        
        return next_state, reward,action, update,terminal

    def get_reward(self, state,next_state):
        prev = np.where(self.state == state)[0][0]
        next_ = np.where(self.state == next_state)[0][0]    
        return self.reward[str(prev)][int(next_)]

    def save(self, save=False, episode=0,action=None):
        if save:
            csvwriter('/home/mtb/rulo_ws/src/rulo_pkg/rulo_ml/data/reward_episode.csv',
                ['episode','reward'],[[episode],[self.all_reward]])
    
            self.actions = np.array([self.cleaned_cells],dtype='int')
            numpywriter('/home/mtb/rulo_ws/src/rulo_pkg/rulo_ml/data/action.txt', self.actions)
        else:
            self.cleaned_cells = [action]
            self.all_reward = self.reward[str(action)][action]
            return self.all_reward
    
    def save_loss(self,loss,episode):
        csvwriter('/home/mtb/rulo_ws/src/rulo_pkg/rulo_ml/data/loss.csv',
                ['episode','loss'],[[episode],[loss]])

# if __name__ == '__main__':
    # a = [25, 27, 8, 48, 0, 24, 28, 34, 6, 3, 37, 15, 45, 17, 26, 49, 1, 36, 35, 32, 33, 16, 7, 42, 20,
    #      43, 23, 41, 13, 44, 46, 29, 40, 30, 10, 22, 47, 18, 2, 19, 21, 38, 11, 9, 12, 14, 5, 39, 31, 4]
    # print len(a)
