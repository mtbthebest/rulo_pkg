#!/usr/bin/env	python
import numpy as np
from collections import deque, OrderedDict
import random

class ReplayBuffer:
    
    def __init__(self, buffer_size,random_seed = 1234):
        self.buffer_size = buffer_size
        self.count = 0
        np.random.seed(random_seed)
    
    def add(self, s, a, r, t, s2):
        experience = [[s, a, r, t, s2]]     
        try:   
            if self.buffer.shape[0] >=self.buffer_size:
                self.buffer = np.delete(self.buffer,0,axis=0)
                self.concat(experience)
            else:
                self.concat(experience)
        except:
            self.concat(experience)
        self.count = self.buffer.shape[0]
        
    def size(self):
        return self.count

    def concat(self,experience):
        try:
            self.buffer = np.concatenate((self.buffer, experience), axis=0)
        except:
            self.buffer = np.array(experience)

    def sample_batch(self, batch_size):
        idx = np.random.randint(low=0, high=self.count, size=batch_size)
        batch = self.buffer[idx,:]    

        s_batch = [elem.tolist() for elem in batch[:,0]]
        a_batch = [elem.tolist() for elem in batch[:, 1]]
        r_batch = batch[:,2]
        t_batch = batch[:, 3] 
        s2_batch = [elem.tolist() for elem in batch[:, 4]]

        return s_batch, a_batch, r_batch, t_batch, s2_batch

