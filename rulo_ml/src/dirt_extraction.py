#!/usr/bin/env	python
import rospy  
import os
import csv
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from collections import deque, OrderedDict
from rulo_utils.csvreader import csvread
from rulo_utils.csvcreater import csvcreater
from rulo_utils.csvwriter import csvwriter
from rulo_utils.csvconverter import csvconverter, convert_list_to_int, convert_lists_to_int
from math import copysign, fabs
from rulo_base.markers import VizualMark, TextMarker
from rulo_base.path_creater import Path
from rulo_utils.graph_plot import Plot
import matplotlib.pyplot as plt
import time
import datetime
from math import pow

path = '/media/mtb/Data Disk/data/dirt/test2017-10-07_13h.csv'

 
corners = [[-2.25, -4.0], [-2.25, 8.0], [4.5, 8.0], [4.5, -4.0]]
grid_size = 0.25

class Process:
    
    def __init__(self):   
        self.corners = OrderedDict()
        self.centers = deque()  
        self.get_rectangle_corners()
        print 'init'
    def get_center(self):
        '''
        Returns a list of centers position list
        ---
        self.centers = [[0.5,0.5],[1.0,1.0],...]
        '''
        x_start, y_start = corners[0][0], corners[0][1]
        self.centers.append(
            [x_start + grid_size / 2.0, y_start + grid_size / 2])
        center_start = self.centers[0]
        list_x, list_y = deque(), deque()
        for [x, y] in corners:
            list_x.append(x)
            list_y.append(y)
        increment = 0.0
        while(self.centers[-1][1] < max(list_y)):
             while(self.centers[-1][0] < max(list_x)):
                 val = [self.centers[-1][0] + grid_size, self.centers[-1][1]]
                 if (val[0] < max(list_x)):
                     self.centers.append(val)
                 else:
                    break
             if self.centers[-1][1] + grid_size < max(list_y):
                 self.centers.append(
                     [self.centers[0][0], self.centers[-1][1] + grid_size])
             else:
                break
        del list(list_x)[:]
        del list(list_y)[:]
        return self.centers

    def get_rectangle_corners(self):
        '''
        Returns a list of centers coordinates and a dictionnary of corners coordinates 
        --- 
        self.corners = {'[0.5,0.5]':[[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]],...}
        '''
        self.corners = OrderedDict()
        
        self.get_center()
        for center in self.centers:
            self.corners[str(center)] = [[center[0] - grid_size / 2, center[1] - grid_size / 2],
                                         [center[0] + grid_size / 2,
                                             center[1] - grid_size / 2],
                                         [center[0] + grid_size / 2,
                                             center[1] + grid_size / 2],
                                         [center[0] - grid_size / 2, center[1] + grid_size / 2]]
        return self.corners

    def get_file_pose_and_dirt_level(self):
        '''
        Returns a list of pose for each axis, dirt level recorded by each sensor
        ---
        self.pose = [[0.25, 0.25],...]                 
        self.dirt_h_level = [280, 282,...]      
        self.dirt_l_level = [280, 282,...]     
        '''
        # self.filename = '/media/mtb/Data Disk/data/dirt/test2017-11-07_12h35.csv'
        data = csvread(self.filename)
        pose = []
        dirt_h_level,  dirt_l_level, wall_time = [],[],[]
        duration = [0.0]
        for i in range(len(data.values()[0])):
            pose.append([float(data['p_x'][i]),float(data['p_y'][i])])
            dirt_h_level.append(int(data['dirt_high_level'][i]))
            dirt_l_level.append(int(data['dirt_low_level'][i]))
            wall_time.append(float(data['wall_time'][i]))
            # try:
            if i>=1:
                
                if fabs(wall_time[-1] - wall_time[-2]) < 2.0:                     
                    duration.append(fabs(wall_time[-1] - wall_time[-2]))
                else: 
                    duration.append(0.0)
            # except:
            #     pass
        # self.start = self.wall_time[0]
        self.pose = np.array(pose)
        self.dirt_h_level = np.array(dirt_h_level)
        self.dirt_l_level = np.array(dirt_l_level)
        self.wall_time = np.array(wall_time)
        self.duration = np.array(duration)
        # print self.dirt_h_level
        # print min(self.duration),max(self.duration)
        # print self.filename, 'start: ', self.start
        # print self.wall_time
        # print len(self.duration), len(self.wall_time)
        # return self.pose, self.dirt_h_level, self.dirt_l_level , self.wall_time

    def get_total_dirt_num(self, key=['h','l'],filename=[], file=[]):
        '''        
        Returns  the pose where dirt was spotted and the number of particles
        ---
        pose= [[0.25, 0.5],......]         
        dirt_num = {key = 'h': [200,...]}                       
        '''       
        # dirt_pose, dirt_level = self.get_dirt_pose_and_dirt_level()
        for t in range(len(filename)):
            self.filename = filename[t]
            self.file = file[t]
            print self.file
            self.get_file_pose_and_dirt_level()
            self.grid_dirt = OrderedDict()
            self.cleaned_time_cycle = OrderedDict()
            self.time_spent_in_cells = OrderedDict()
            self.dirt_sucked_in_time = OrderedDict()
            self.dirt_sucked_val = OrderedDict()
            for level in key:
                self.grid_dirt[level] =OrderedDict()
                # self.cleaned_time_cycle[level] = OrderedDict()
                self.dirt_sucked_in_time[level] = OrderedDict()
                self.dirt_sucked_val[level] = OrderedDict()
                for key_center in self.corners.keys():
                    self.grid_dirt[level][key_center] = 0
                    self.cleaned_time_cycle[key_center] =0.0
                    self.time_spent_in_cells[key_center] = 0.0
                    self.dirt_sucked_in_time[level][key_center] = [0.0]
                    self.dirt_sucked_val[level][key_center] = [0]
            # print 'Starting the operation'
            for k in range(self.pose.shape[0]):
                time_update = False
                if k ==0:
                    self.dirt_high_sensor, self.dirt_low_sensor = self.dirt_h_level[0], self.dirt_l_level[0]
                else:
                    if self.dirt_h_level[k] !=self.dirt_high_sensor:
                        val = self.dirt_h_level[k] - self.dirt_high_sensor
                        if val <0:
                            val +=256
                        cell , update= self.update_pose(self.pose[k])
                        if update: 
                            self.grid_dirt['h'][cell] +=val
                            self.cleaned_time_cycle[cell] = self.wall_time[k]
                            self.time_spent_in_cells[cell] +=self.duration[k]
                            self.dirt_sucked_in_time['h'][cell].append(self.time_spent_in_cells[cell])
                            self.dirt_sucked_val['h'][cell].append(val)

                            time_update =True
                            # self.cleaned_time_cycle['h'][cell] = self.wall_time[k]
                        self.dirt_high_sensor = self.dirt_h_level[k]

                    if self.dirt_l_level[k] != self.dirt_low_sensor:
                        val = self.dirt_l_level[k] - self.dirt_low_sensor
                        if val < 0:
                            val += 256
                        cell, update = self.update_pose(self.pose[k])
                        if update:
                            self.grid_dirt['l'][cell] += val  
                            self.cleaned_time_cycle[cell] = self.wall_time[k]
                            self.time_spent_in_cells[cell] += self.duration[k]
                            self.dirt_sucked_in_time['l'][cell].append(self.time_spent_in_cells[cell])
                            self.dirt_sucked_val['l'][cell].append(val)
                            time_update = True
                            # self.cleaned_time_cycle['l'][cell] = self.wall_time[k]
                        self.dirt_low_sensor = self.dirt_l_level[k]
                    
                    if not time_update:
                        cell, update = self.update_pose(self.pose[k])
                        if update:
                            self.cleaned_time_cycle[cell] = self.wall_time[k]
                            self.time_spent_in_cells[cell] += self.duration[k]
            
            csvwriter('/home/mtb/Documents/data/dirt_extraction_2/data/high/dirt_value/' + self.file,
                        headers=['wall_time', 'dirt_level','cleaning_duration'],
                    rows=[self.cleaned_time_cycle.values(),self.grid_dirt['h'].values(), self.time_spent_in_cells.values()])
            
            np.save('/home/mtb/Documents/data/dirt_extraction_2/data/high/time/' +\
                    self.file[:-4] + '.npy', np.array(self.dirt_sucked_in_time['h'].values()))
            np.save('/home/mtb/Documents/data/dirt_extraction_2/data/high/dirt_sucked_by_cells/' +\
                    self.file[:-4] + '.npy', np.array(self.dirt_sucked_val['h'].values()))
            csvwriter('/home/mtb/Documents/data/dirt_extraction_2/data/low/dirt_value/' + self.file,
                      headers=['wall_time', 'dirt_level', 'cleaning_duration'],
                      rows=[self.cleaned_time_cycle.values(), self.grid_dirt['l'].values(), self.time_spent_in_cells.values()])
            np.save('/home/mtb/Documents/data/dirt_extraction_2/data/low/time/' +\
                    self.file[:-4] + '.npy', np.array(self.dirt_sucked_in_time['l'].values()))
            np.save('/home/mtb/Documents/data/dirt_extraction_2/data/low/dirt_sucked_by_cells/' +\
                    self.file[:-4] + '.npy', np.array(self.dirt_sucked_val['l'].values()))
            # del self.pose
            # del self.dirt_h_level
            # del self.dirt_l_level
            # del self.grid_dirt
            # del self.wall_time
            # del self.duration
            for val in[self.pose, self.dirt_h_level, self.dirt_l_level, self.wall_time, self.duration]:
                self.reset(val)
    
    def update_pose(self, position):
        finish = False
        for key_center in self.corners.keys():
                x_list = []
                y_list = []
                for value in self.corners[key_center]:
                    x_list.append(value[0])
                    y_list.append(value[1])
                x_min = min(x_list)
                x_max = max(x_list)
                y_min = min(y_list)
                y_max = max(y_list)
                if (position[0] > x_min and position[0] < x_max) and (position[1] > y_min and position[1] < y_max):
                    finish = True
                    return key_center, finish
                    break
        if not finish:
            return key_center, finish

    
    def reset(self, value):
      
            value = np.delete(value, range(0, value.shape[0]))
            return value
     


if __name__ == '__main__': 

    filename_list = []
    file_list = []
    # path = '/media/mtb/Data Disk/data/dirt/'
    # for files in os.listdir(path ):
    #     filename_list.append(path + files)
    #     file_list.append(files)
    # for step in range(21,len(filename_list)):
    Process().get_total_dirt_num(key=['h','l'], filename =['/home/mtb/Documents/data/test2017-02-19_high.csv'], file=['test2017-02-19_high.csv'])
    # Process().get_file_pose_and_dirt_level()
    # a = np.array([[]])
    # a = np.concatenate((a,[[1,2]]), axis=1)
    # a = np.concatenate((a, [[1, 2],[2,4]]))
    # print a
