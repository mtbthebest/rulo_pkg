#!/usr/bin/env	python
import rospy 
import os
import sys
import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer
import numpy as np
from rulo_base.markers import VizualMark, TextMarker
import matplotlib.pyplot  as plt
from collections import OrderedDict
from rulo_utils.graph_plot import Plot
import pandas as pd
from pandas import DataFrame, Series
from Memory import Memory 
from rulo_utils.csvwriter import csvwriter
import random
from time import sleep


class RL:

    def __init__(self):
        self.data = pd.read_csv('/home/mtb/rulo_ws/src/rulo_pkg/rulo_ml/dqn_server/02-05/4/summary/summary.csv')
    
    def plot(self):
        self.episodes = self.data['episodes']
        self.explore = self.data['explore']
        self.cost_function = self.data['cost_function']
        self.simulation_reward = self.data['episode_reward']
        self.wander = self.data['wander']
        
        self.total_rewards = self.data['total_rewards']
        save_path = '/home/mtb/rulo_ws/src/rulo_pkg/rulo_ml/dqn_server/02-05/4/'
        limit =100
        start = 0
        # print list(np.sort(self.simulation_reward))
        Plot().scatter_plot(list(self.episodes.values[start::limit]), list(self.cost_function.values[start::limit]), marker='r-',title='Cost_function' ,   labels =['Episodes', 'Cost function'],save_path=save_path + 'cost_function.png')
        Plot().scatter_plot(list(self.episodes.values[start::limit]), list(self.explore[start::limit]), marker='r-',title='Exploration' ,   labels =['Episodes', 'Exploration'],save_path=save_path + 'exploration.png')
        Plot().scatter_plot(list(self.episodes.values[start::limit]), list(self.simulation_reward[start::limit]), marker='r-',title='Policy reward' ,   labels =['Episodes', 'reward'],save_path=save_path + 'policy_reward.png')
        Plot().scatter_plot(list(self.episodes.values[start::limit]), list(self.wander[start::limit]), marker='r-',title='Random Tries by episode' ,   labels =['Episodes', 'Random tries'],save_path=save_path + 'random_tries.png')
        Plot().scatter_plot(list(self.episodes.values[start::limit]), list(self.total_rewards)[start::limit], marker='r-',title='Total Rewards per episode' ,   labels =['Episodes', 'Reward'],save_path=save_path + 'episode_reward.png')
    
    def visualize(self):
        self.cells = self.data['cells']
        data_cells = self.cells.iloc[-1]
        cells = []

        for elem in data_cells[1:-1].split(','):
            try:
                cells.append(int(elem))
            except:
                pass       
        # self.states = np.load('/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/data/pose.npy')
        # self.dirt_lev= np.load('/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/data/dirt.npy')
        
if __name__ == '__main__':
    RL().plot()
    