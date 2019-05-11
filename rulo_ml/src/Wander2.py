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
        # if not os.path.isfile('/home/mtb/rulo_ws/src/rulo_pkg/rulo_ml/data/rewards_dqn_high_375_623.csv'):
        self.time = pd.read_csv('/home/mtb/rulo_ws/src/rulo_pkg/rulo_ml/data/cells_time.csv', index_col=[0])
        print self.time   
        limit_inf =375
        limit_sup = 623
        self.reward = self.time.iloc[limit_inf:limit_sup, limit_inf:limit_sup]
        self.dirt_lev= np.load('/home/mtb/rulo_ws/src/rulo_pkg/rulo_ml/dqn/data/dirt_high_19.npy')    
        # print list(self.dirt_lev)
        for i in range(limit_inf,limit_sup):
            for j in range(limit_inf,limit_sup):
                if i == j:
                    self.reward[str(i)][j] = float(self.dirt_lev[i])
                else:
                    # print float(self.dirt_lev[j]), self.reward[str(i)][j] 
                    self.reward[str(i)][j] = float(self.dirt_lev[j])/ (self.reward[str(i)][j] + 8.0)
                    # print self.reward[str(i)][j]
        max_ =  np.amax(self.reward.max().values)      
        self.reward.to_csv('/home/mtb/rulo_ws/src/rulo_pkg/rulo_ml/data/rewards_dqn_high_375_623.csv')
        # else:
        #     self.reward = pd.read_csv('/home/mtb/rulo_ws/src/rulo_pkg/rulo_ml/data/rewards_dqn_high_375_623.csv', index_col=[0])
        #     max_ =  np.amax(self.reward.max().values)  
        #     print max_    
    
        # self.data = pd.read_csv('/home/mtb/rulo_ws/src/rulo_pkg/rulo_ml/dqn/data/dqn_data_state_dirt_high_375_623.csv', index_col=[0]) 
               
        # self.positions = self.rewrite(columns='positions', type='float')
        # self.possibility = self.rewrite(columns='possibility', type='float')
        # self.states = list(self.data.iloc[:].index.values)
        # # print len(self.states)
        # self.input_states = []
        # for i in range(len(self.states)):
        #     cell_val = self.states[i] / max(self.states)
        #     self.input_states.append([ cell_val,0.0])

    def initialize(self):
        # num = 24
        # return self.state[num].reshape([1,2])
        pass

    def reset(self):
        self.wandered = []
        start_cell =144       
        self.all_states = self.input_states[:]
        self.all_states[start_cell][1] = 1.0
        s_t_d, s_t_p = np.array(self.all_states)[:,0 ], np.array(self.all_states)[:,1 ]
        return [s_t_d,s_t_p]
      
    def step(self, present_position, s_t_d, s_t_p, action):
        new_state_d = np.copy(s_t_d)
        new_state_p = np.copy(s_t_p)
        new_position = int(self.positions[present_position][action])
        if self.possibility[present_position][action] == 1.0:
            if new_position not in self.wandered:
                reward = self.get_reward(present_position, new_position)
                self.wandered.append(new_position)
            else:
                reward = -10.0
            new_state_d[new_position] = 0.0
            new_state_p[present_position] = 0.0
            new_state_p[new_position] = 1.0
            terminal = False
        else:
            reward = -100.0
            terminal = True
        return new_state_d, new_state_p, reward, new_position, terminal

    def get_reward(self, prev,new):   
        return self.reward[str(prev)][int(new)]

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
    
    def rewrite(self, columns='action', type='int'):
        res = []
        for elem1 in self.data[columns].values:
            if type == 'int':
                trans = map(int, elem1[1:-1].split(','))
                res.append(trans)
            if type == 'float':
                frac = elem1[1:-1].split(',')
                trans = []
                for val in frac:
                    try:
                        trans.append(float(val))
                    except:
                        pass
                res.append(trans)
        return res
    

if __name__ == '__main__':
   Wander()
