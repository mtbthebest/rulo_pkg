#!/usr/bin/env python

import os
import numpy as np
from collections import deque, OrderedDict
from rulo_utils.csvreader import csvread
from rulo_utils.csvwriter import csvwriter
from rulo_utils.graph_plot import Plot
import matplotlib.pyplot as plt

path = '/home/mtb/Documents/data/test2017-12-28_inplace.csv'


corners = [[-2.25, -4.0], [-2.25, 8.0], [4.5, 8.0], [4.5, -4.0]]
grid_size = 0.25
dirt_size = 0.02

report_filename = '/home/mtb/Documents/data/12-4/'
cleaned_cell_filename = '/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/data/pose.csv'


class Process:
    def __init__(self, id_num=None):
        self.dirt_h_level,  self.dirt_l_level, self.wall_time = deque(), deque(), deque()
        self.filename = path

    def read_file(self):
        '''
        Returns a list of pose for each axis, dirt level recorded by each sensor
        ---
        self.pose = [[0.25, 0.25],...]                 
        self.dirt_h_level = [280, 282,...]      
        self.dirt_l_level = [280, 282,...]     
        '''
        self.data = csvread(self.filename)
        for i in range(len(self.data.values()[0])):
            self.dirt_h_level.append(int(self.data['dirt_high_level'][i]))
            self.dirt_l_level.append(int(self.data['dirt_low_level'][i]))
            self.wall_time.append(float(self.data['wall_time'][i]))
          
        return self.dirt_h_level, self.dirt_l_level, self.wall_time

    def get_dirt_level(self,key ='h'):
        '''        
        Returns  the pose where dirt was spotted and the number of particles
        ---
        dirt_pose={'h': [[0.25, 0.5],...],...], 'l':[[0.25, 0.5],...],...]}         
        dirt_level = {'h': [200,...], 'l': [200,...]}                       
        '''
        self.read_file()    
        dirt_level =  deque()
        wall_time = deque()
        if key =='h':
            dirt = self.dirt_h_level
        elif key =='l':
            dirt = self.dirt_l_level
        for indice in range(len(self.dirt_h_level)):
            if indice > 0:
                if dirt[indice] !=dirt[indice-1]:
                    val = dirt[indice] - dirt[indice-1]
                    if val <0:
                        val = 256 + val
                    dirt_level.append(val)
                    wall_time.append(self.wall_time[indice])
            else:
                dirt_level.append(0)
                wall_time.append(self.wall_time[indice])
    
        csvwriter('/home/mtb/low.csv',
                 headers=['dirt_high_level', 'wall_time'],
                 rows=[dirt_level, wall_time])

if __name__ == '__main__':

    Process().get_dirt_level(key ='l')
