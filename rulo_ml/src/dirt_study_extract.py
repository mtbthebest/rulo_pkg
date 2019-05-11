#!/usr/bin/env	python

import os
import numpy as np
from collections import deque, OrderedDict
from rulo_utils.csvreader import csvread
from rulo_utils.csvcreater import csvcreater
from rulo_utils.csvwriter import csvwriter
from rulo_utils.csvconverter import csvconverter, convert_list_to_int, convert_lists_to_int

from rulo_utils.graph_plot import Plot
import matplotlib.pyplot as plt
import time
import datetime
from math import pow

# path = '/home/mtb/Documents/data/'
path = '/home/mtb/Documents/data/lin_rot_pwm_var/2017-12-18/'
lin_var_filename = sorted([files for files in os.listdir(path) if files.startswith('pwm_40_lin_')])
# print len(lin_var_filename)

ang_var_filename = sorted([files for files in os.listdir(path) if files.startswith('pwm_40_rot_')])
print ang_var_filename
pwm_var_filename = ['pwm_var_30_rot_0.3.csv',
                    'pwm_var_50_rot_0.3.csv', 'pwm_var_60_rot_0.3.csv']
                    
corners = [[-2.25, -4.0], [-2.25, 8.0], [4.5, 8.0], [4.5, -4.0]]
grid_size = 0.25
dirt_size = 0.02
max_dirt = 20000
color = {'low': 'Blue', 'medium': 'Yellow', 'high': 'Red'}

grid_dirt_filename = '/home/mtb/Documents/data/train/rnn_data/'
train_file_saving_path = '/home/mtb/Documents/data/features/11-24/'
cell_num = 1296
class_num = 20

report_filename = '/home/mtb/Documents/data/12-4/'
cleaned_cell_filename = '/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/data/pose.csv'


class Process:
    def __init__(self, id_num=None):
        self.pose = deque()
        self.dirt_h_level,  self.dirt_l_level, self.wall_time,self.linear_vel, self.angular_vel, self.pwm= deque(), deque(), deque(), deque(), deque(), deque()
        self.laser_ranges = deque()
        self.corners = OrderedDict()
        self.centers = deque()
        self.time = deque()
        self.grid_point = OrderedDict()
        self.labels = deque()
        self.input = deque()
        # if id_num:
        self.file_id = id_num
        print self.file_id
        

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
        self.data = csvread(path + self.filename)
        for i in range(len(self.data.values()[0])):
            # self.pose.append([float(self.data['pose_x'][i]),
            #                   float(self.data['pose_y'][i])])
            # self.time.append(float(self.data['wall_time'][i]))
            self.dirt_h_level.append(int(self.data['dirt_high_level'][i]))
            self.dirt_l_level.append(int(self.data['dirt_low_level'][i]))
            # self.linear_vel.append(float(self.data['lin_x'][i]))
            # self.angular_vel.append(float(self.data['rot_z'][i]))
            # self.pwm.append(int(self.data['main_brush'][i]))

            #self.wall_time.append(float(self.data[i]['wall_time']))
        # self.test_time =   time.localtime(self.wall_time[0] * pow(10, -9) )[1:6]
        # print self.time
        return self.dirt_h_level, self.dirt_l_level#, self.linear_vel, self.angular_vel  # , self.wall_time

    def get_dirt_pose_and_dirt_level(self):
        '''        
        Returns  the pose where dirt was spotted and the number of particles
        ---
        dirt_pose={'h': [[0.25, 0.5],...],...], 'l':[[0.25, 0.5],...],...]}         
        dirt_level = {'h': [200,...], 'l': [200,...]}                       
        '''
        self.get_file_pose_and_dirt_level()
        dirt_level = OrderedDict()
        # dirt_pose['h'] = deque()
        # dirt_pose['l'] = deque()
        dirt_level['h'] = deque()
        dirt_level['l'] = deque()

        for indice in range(len(self.dirt_h_level)):
            if indice > 0:
                if self.dirt_h_level[indice] != dirt_level['h'][-1]:
                    # dirt_pose['h'].append(self.pose[indice])
                    dirt_level['h'].append(self.dirt_h_level[indice])
                if self.dirt_l_level[indice] != dirt_level['l'][-1]:
                    # dirt_pose['l'].append(self.pose[indice])
                    dirt_level['l'].append(self.dirt_l_level[indice])
            else:
                # dirt_pose['h'].append(self.pose[indice])
                # dirt_pose['l'].append(self.pose[indice])
                dirt_level['h'].append(self.dirt_h_level[indice])
                dirt_level['l'].append(self.dirt_l_level[indice])

        return dirt_level

    def get_total_dirt_num(self, key='h'):
        '''        
        Returns  the pose where dirt was spotted and the number of particles
        ---
        pose= [[0.25, 0.5],......]         
        dirt_num = {key = 'h': [200,...]}                       
        '''
        dirt_level = self.get_dirt_pose_and_dirt_level()
        dirt_num = OrderedDict()
        dirt_num[key] = deque()
        dirt_num[key].append(0)
        # pose = []
        # pose.append(dirt_pose[key][0])
        for i in range(1, len(dirt_level[key])):
                dirt_value = dirt_level[key][i] - dirt_level[key][i - 1]
                if dirt_value < 0:
                    dirt_value += 256
                # if dirt_pose[key][i] in pose:
                #     dirt_num[key][pose.index(dirt_pose[key][i])] += dirt_value
                dirt_num[key].append(dirt_value)
                    # pose.append(dirt_pose[key][i])
        return dirt_num

    

    def get_linear_dirt(self, key = 'h'):
        self.linear_dirt = []
        # self.filename = lin_var_filename[0]
        # print  (self.get_total_dirt_num(key)[key])
        for i in range(len(lin_var_filename)):
            self.file_id = i
            self.filename = lin_var_filename[i]
            self.linear_dirt.append(sum(list(self.get_total_dirt_num(key)[key])))
      

        # print len(self.linear_dirt)
        x = [2 * int(i+1) for i in range(len(lin_var_filename))]
        # print 
    
        Plot().plot_bar(x= x, y=self.linear_dirt,x_axis ='linear_velocity', y_axis ='total_dirt_recorded', x_labels= [0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.12,0.14,0.16,0.18,0.20])
    
    def get_angular_dirt(self, key = 'h'):
        self.angular_dirt = []
        for i in range(len(ang_var_filename)):
            self.file_id = i
            self.filename = ang_var_filename[i]
            self.angular_dirt.append(sum(list(self.get_total_dirt_num(key)[key])))
        # print self.pwm[0]
        print self.angular_dirt

        x = [2 * int(i+1) for i in range(len(ang_var_filename))]
        Plot().plot_bar(x=x, y=self.angular_dirt, x_axis='angular_velocity',
                         y_axis='total_dirt_recorded', width=1, x_labels=[0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])

    def get_pwm_dirt(self, key='h'):
        self.pwm_dirt = []
        for i in range(len(pwm_var_filename)):
            self.file_id = i
            self.filename = pwm_var_filename[i]
            self.pwm_dirt.append(
                sum(list(self.get_total_dirt_num(key)[key])))
        print self.pwm[0]
        print self.pwm_dirt

        Plot().plot_bar(x=[0.1,0.2,0.3], y=self.pwm_dirt, x_axis='pwm_speed',
                        y_axis='total_dirt_recorded', width=0.05, x_labels=[30, 50,60])

    
    def plot_dirt_cleaning_in_time(self, key='h'):        
        dirt_number = self.get_total_dirt_num(key)[key]
        
    




if __name__ == '__main__':

    Process().get_angular_dirt('l')
   