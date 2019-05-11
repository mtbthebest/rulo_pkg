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

# path = '/home/mtb/Documents/data/'
path = '/media/mtb/Data Disk/data/dirt/'
# filename = ['test2017-10-04_11h_45.csv', 'test2017-10-05_16h.csv', 'test2017-10-06_11h_15.csv', 'test2017-10-06_21h.csv',
#             'test2017-10-07_13h.csv', 'test2017-10-07_18h25.csv', 'test2017-10-10_08h20.csv', 'test2017-10-10_19h.csv',
#             'test2017-10-11_12h15.csv', 'test2017-10-11_19h.csv', 'test2017-10-12_11h20.csv', 'test2017-10-13_14h.csv',
#             'test2017-10-13_21h.csv', 'test2017-10-16_10h45.csv', 'test2017-10-16_18h30.csv', 'test2017-10-17_10h30.csv',
#             'test2017-10-17_18h30.csv', 'test2017-10-18_12h20.csv', 'test2017-10-18_19h10.csv', 'test2017-10-19_13h10.csv',
#             'test2017-10-20_12h30.csv', 'test2017-10-20_19h30.csv', 'test2017-10-24_11h30.csv', 'test2017-10-24_18h40.csv',
#             'test2017-10-25_11h00.csv', 'test2017-10-26_10h45.csv', 'test2017-11-07_12h35.csv',  'test2017-11-08_11h15.csv',
#             'test2017-11-10_11h40.csv', 'test2017-11-14_11h40.csv', 'test2017-11-16_12h00.csv', 'test2017-11-17_12h40.csv',
#             'test2017-11-18_15h00.csv', 'test2017-11-21_11h45.csv', 'test2017-11-22_14h00.csv', 'test2017-11-23_13h00.csv',
#             'test2017-11-28_12h15.csv', 'test2017-11-29_10h20.csv', 'test2017-11-30_10h55.csv', 'test2017-12-02_12h30.csv',
#             'test2017-12-04_09h00.csv', 'test2017-12-05_12h25.csv', 'test2017-12-07_10h15.csv', 'test2017-12-09_11h10.csv'
#             ]
 
corners = [[-2.25, -4.0], [-2.25, 8.0], [4.5, 8.0], [4.5, -4.0]]
grid_size = 0.25
dirt_size = 0.02
max_dirt = 20000
color = {'low': 'Blue', 'medium': 'Yellow', 'high': 'Red'}

grid_dirt_filename = '/home/mtb/Documents/data/train/rnn_data/'
train_file_saving_path = '/home/mtb/Documents/data/features/11-24/'
cell_num = 1296
class_num = 20
cleaning_cycle = OrderedDict([
            ('rnn_train_2.csv',  0),
            ('rnn_train_11.csv',  120),
            ('rnn_train_12.csv',  7),
            ('rnn_train_13.csv', 13),
            ('rnn_train_14.csv', 8),
            ('rnn_train_15.csv',  15),
            ('rnn_train_17.csv',  12),
            ('rnn_train_18.csv',  7),
            ('rnn_train_19.csv',  18),
            ('rnn_train_20.csv', 24),
            ('rnn_train_21.csv',  7),
            ('rnn_train_22.csv',  87),
            ('rnn_train_23.csv',  7),
            ('rnn_train_24.csv',  16),
            ('rnn_train_25.csv',  24),
            ('rnn_train_26.csv',  290),
            ('rnn_train_27.csv',  24),
            ('rnn_train_28.csv',  24),     
            ('rnn_train_29.csv',  96),
            ('rnn_train_30.csv',  48),
            ('rnn_train_32.csv',  51)
            
           
            ])
            #('rnn_train_33.csv',  68),
            # ('rnn_train_34.csv',  27),
            # ('rnn_train_35.csv',  23)
# ('rnn_train_32.csv',  27)
report_filename = '/home/mtb/Documents/data/12-4/'
cleaned_cell_filename = '/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/data/pose.csv'

class Process:
    def __init__(self, filename=None,files=None):
        self.pose = deque()
        self.dirt_h_level,  self.dirt_l_level, self.wall_time = deque(), deque(), deque()
        self.laser_ranges = deque()
        self.corners = OrderedDict()
        self.centers = deque()

        self.grid_point = OrderedDict()
        self.labels = deque()
        self.input = deque()
     
        self.filename = filename
      
        self.file = files
        # if id_num:
        #     self.file_id = id_num
        #     print self.file_id
            
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
        # np.save('/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/data/all_cells_pose.csv',np.array(self.centers))
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
        self.data = csvread(self.filename)
        for i in range(len(self.data.values()[0])):
            self.pose.append([float(self.data['p_x'][i]),
                              float(self.data['p_y'][i])])
            self.dirt_h_level.append(int(self.data['dirt_high_level'][i]))
            self.dirt_l_level.append(int(self.data['dirt_low_level'][i]))
            #self.wall_time.append(float(self.data[i]['wall_time']))
        # self.test_time =   time.localtime(self.wall_time[0] * pow(10, -9) )[1:6]
        return self.pose, self.dirt_h_level, self.dirt_l_level  # , self.wall_time

    def get_dirt_pose_and_dirt_level(self):
        '''        
        Returns  the pose where dirt was spotted and the number of particles
        ---
        dirt_pose={'h': [[0.25, 0.5],...],...], 'l':[[0.25, 0.5],...],...]}         
        dirt_level = {'h': [200,...], 'l': [200,...]}                       
        '''
        self.get_file_pose_and_dirt_level()
        dirt_pose, dirt_level = OrderedDict(), OrderedDict()
        dirt_pose['h'] = deque()
        dirt_pose['l'] = deque()
        dirt_level['h'] = deque()
        dirt_level['l'] = deque()

        for indice in range(len(self.pose)):
            if indice > 0:
                if self.dirt_h_level[indice] != dirt_level['h'][-1]:
                    dirt_pose['h'].append(self.pose[indice])
                    dirt_level['h'].append(self.dirt_h_level[indice])
                if self.dirt_l_level[indice] != dirt_level['l'][-1]:
                    dirt_pose['l'].append(self.pose[indice])
                    dirt_level['l'].append(self.dirt_l_level[indice])
            else:
                dirt_pose['h'].append(self.pose[indice])
                dirt_pose['l'].append(self.pose[indice])
                dirt_level['h'].append(self.dirt_h_level[indice])
                dirt_level['l'].append(self.dirt_l_level[indice])

        return dirt_pose, dirt_level

    def get_total_dirt_num(self, key='h'):
        '''        
        Returns  the pose where dirt was spotted and the number of particles
        ---
        pose= [[0.25, 0.5],......]         
        dirt_num = {key = 'h': [200,...]}                       
        '''
        dirt_pose, dirt_level = self.get_dirt_pose_and_dirt_level()
        dirt_num = OrderedDict()
        dirt_num[key] = deque()
        dirt_num[key].append(0)
        pose = []
        pose.append(dirt_pose[key][0])
        for i in range(1, len(dirt_level[key])):
                dirt_value = dirt_level[key][i] - dirt_level[key][i - 1]
                if dirt_value < 0:
                    dirt_value += 256
                if dirt_pose[key][i] in pose:
                    dirt_num[key][pose.index(dirt_pose[key][i])] += dirt_value
                else:
                    dirt_num[key].append(dirt_value)
                    pose.append(dirt_pose[key][i])
        return pose, dirt_num

    def get_room_cleaned_centers(self, key='h'):
        self.get_grid_dirt(key)
        self.cleaned_cells = OrderedDict()
        # center_cells =csvread(cleaned_cell_filename)['pose']
        for values in csvread(cleaned_cell_filename)['pose']:
            # self.cleaned_cells[values] = self.grid_dirt[values]
            csvwriter(report_filename + '12-9' + key + '.csv', ['pose', 'dirt_num'],
                 [[values], [self.grid_dirt[values]]])
        # print self.cleaned_cells.keys()
        # print self.cleaned_cells.values()
        return self.cleaned_cells
        # print len(csvread(cleaned_cell_filename)['pose'])

    def visualize_cleaned_dirt(self, ):
        cleaned_file_read = csvread(report_filename + '12-9_l.csv')
        pose = []
        for keys in cleaned_file_read['pose']:
            position =[]
            for elem in keys[1:-1].split(','):
                position.append(float(elem))
            pose.append(position)
        color_dict ={'Red':10000,
                    'Yellow':5000,
                    'Orange':2500,
                    'Green':1000,
                    'Blue':500}
        color_list = []
     
        for position in pose:
            indice = cleaned_file_read['pose'].index(str(position))
            print cleaned_file_read['dirt_num'][indice]
            if int(cleaned_file_read['dirt_num'][indice]) >= color_dict['Red']:
                color_list.append('Red')
            elif color_dict['Yellow']<= int(cleaned_file_read['dirt_num'][indice]) < color_dict['Red']:
                 color_list.append('Yellow')
            elif color_dict['Orange'] <= int(cleaned_file_read['dirt_num'][indice]) < color_dict['Yellow']:
                 color_list.append('Orange')
            elif color_dict['Green'] <= int(cleaned_file_read['dirt_num'][indice]) < color_dict['Orange']:
                 color_list.append('Green')
            elif color_dict['Blue'] <= int(cleaned_file_read['dirt_num'][indice]) < color_dict['Green']:
                 color_list.append('Blue')
            elif 0 < int(cleaned_file_read['dirt_num'][indice]) < color_dict['Blue']:
                color_list.append('Gray')
            else:
                color_list.append('White')
        size_list = [[grid_size, grid_size,0.0]] * len(color_list)
     

        VizualMark().publish_marker(pose= pose,sizes= size_list, color=color_list)

    def get_grid_dirt(self, key='h'):
        '''
        Return a dictionary of dirt level in every cell of the map
        ---
        self.grid_dirt =  {'[0.5,0.5]': 120,..........}
        '''
        self.grid_dirt = OrderedDict()
        pose, dirt = self.get_total_dirt_num(key)
        dirt_list = list(dirt[key])  
        self.get_rectangle_corners()            
        for key_center in self.corners.keys():              
                x_list = []
                y_list = []
                
                self.grid_dirt[key_center] = 0
                for value in self.corners[key_center]:
                    x_list.append(value[0])
                    y_list.append(value[1])
                x_min = min(x_list)
                x_max = max(x_list)
                y_min = min(y_list)
                y_max = max(y_list)                              
                for [x_position, y_position] in pose:
                    if (x_position > x_min and x_position < x_max) and (y_position > y_min and y_position < y_max):                      
                        index = pose.index([x_position, y_position])                       
                        self.grid_dirt[str(key_center)] += dirt_list[index]       
        # print self.grid_dirt      
        csv_path = '/media/mtb/Data Disk/data/extraction/low level sensor/'  + self.file[0:-4] + '.csv'
        csvwriter(csv_path, ['dirt_level'], [self.grid_dirt.values()]) 
        # return self.grid_dirt        
    
    def get_class(self, key = 'h', cell_val = None, multiclassing = True):        
        if multiclassing:
            self.pose_by_class = OrderedDict()
            for i in range(0, max_dirt,  int(max_dirt/class_num)):      
                self.pose_by_class['class_' + str(int(i)/int(max_dirt/class_num) +1)] = []
            
            interval_step = float(int(max_dirt) / int(class_num)) / float(max_dirt)
            interval_limit = []
            for i in range(class_num):
                value =round(i * interval_step, 6)
                interval_limit.append(value)
        
            for key_center, value_dirt in self.grid_dirt.items():
                prob = float(value_dirt) / float(max_dirt)
                class_ = ''
                if prob >=1.0:
                    self.pose_by_class['class_'  + str(len(interval_limit))].append(key_center)
                else:
                    for val in interval_limit[-1:-len(interval_limit)-1:-1]:
                        if prob>=val:
                            self.pose_by_class['class_' + str(interval_limit.index(val) + 1)].append(key_center)
                            break              
            return self.pose_by_class
        else:
            prob = float(cell_val) / float(max_dirt)
            if 0.010<=prob :
                return np.array([1.0,0.0])
            else:
                return np.array([0.0,1.0])

    def get_data(self):        
        self.cell_dirt = OrderedDict()     
        for file in os.listdir(grid_dirt_filename):    
            self.cell_dirt[file] = deque()         
            data = csvconverter(csvread(os.path.abspath(grid_dirt_filename + file)), 'int')
            for keys in data.keys():
                for grid_dirt in data[keys]:                        
                    self.cell_dirt[file].append(self.get_class(cell_val= float(grid_dirt), multiclassing=False))
        self.input_keys = []
        for key , value in OrderedDict(cleaning_cycle).items():
            self.input_keys.append(key)            
        self.get_features(mode='train')
        self.get_features(mode='test')    
        
    def get_features(self, mode='train'):
        cell_dirt_ordered = OrderedDict()
        labels = OrderedDict()
        if mode == 'train': 
            start_offset = 0
            stop_offset = -1
        if mode =='test':
            start_offset = 1
            stop_offset = len(self.input_keys)        
        file_list_input = list(self.input_keys)[start_offset:stop_offset -1]
        file_list_output = list(self.input_keys)[stop_offset-1]
        print file_list_input
        print file_list_output
        print len(file_list_input)
        for i in range(cell_num):
            cell_dirt_ordered[str(i)] = []            
            for key in file_list_input:                 
                    cell_dirt_ordered[str(i)].append(self.cell_dirt[key][i])
        
            labels[str(i)]=[]
            labels[str(i)].append(self.cell_dirt[file_list_output][i])      
        time_step_list = deque()
        time_step_array_shape = 0
        for files in file_list_input:
            duration_val = cleaning_cycle[files]
            duration = bin(duration_val)[2:]
            bit_list =[]
            for bits in duration:
                bit_list.append(int(bits))
            time_step_array_shape = max(time_step_array_shape, len(bit_list))            
            time_step_list.append(bit_list)       
      
        for array in time_step_list:
            while len(array) < time_step_array_shape:
                array.insert(0, 0)         
        input_list = []
        output_list =[]
        for j in range(cell_num):                            
            for i in range(len(time_step_list)):
                try:
                    cell= np.vstack((cell, np.hstack((np.array(cell_dirt_ordered[str(j)][i]), np.array(time_step_list[i])))))                   
                except:                                       
                    cell= np.hstack((np.array(cell_dirt_ordered[str(j)][i]), np.array(time_step_list[i])))              
            output_list.append(labels[str(j)])
            input_list.append(np.copy(cell))           
            cell = cell.fill(0.0)       
        features_array = np.asarray(input_list) 
        output_array = np.array(output_list)      
        labels_array = output_array.reshape((1296,2))
        print 'Saving ' + mode + ' data...'
        np.save(train_file_saving_path + mode + '_features', np.array(features_array))
        np.save(train_file_saving_path + mode + '_labels'  , labels_array)
        print mode + ' data' + ' Saved.'   
        print features_array.shape
        print labels_array.shape

    def write_grid_dirt_to_file(self):
        file = grid_dirt_filename + 'rnn_train_' + str(self.file_id) + '.csv'
        headers = ['grid_dirt']
        self.get_grid_dirt()
        csvwriter(file,headers,[self.grid_dirt.values()])
        print 'Finished writing'

    def classify_cell(self,key='h'):
        '''
        Classify the pose of the cell in the corresponding cell class
        ---
        self.pose_by_class={'class_1'=[[0.5,0.5], ......], ......}        
        '''
        self.get_grid_dirt(key)        
        self.get_class(key)
        return self.pose_by_class
       
        # for i in range(class_num):
        #     key_class = 'class_' + str(i+1)
        #     self.pose_by_class[key_class] = deque()
        # for key_center in self.centers:
        #     _,class_type = self.get_class(float(self.grid_dirt[str(key_center)])/float(dirt_level_threshold))
        #     self.pose_by_class[class_type].append(key_center)
        # return self.pose_by_class[class_type]      

    def visualize_grid_dirt(self, key='h'):
        '''
        Visualize every cell's dirt level in rviz
        ---
        Returns None
        '''       
        dirt_grid_color = {
            'class_1': 'Aqua',
            'class_2': 'Orange', 
            'class_3': 'Gray', 
            'class_4': 'Pink',
            'class_5': 'Blue',
            'class_6': 'Maroon',
            'class_7': 'Navy',
            'class_8': 'Red',
            'class_9': 'Yellow',
            'class_10': 'Red'
        } 
        self.classify_cell(key)
        for key_class in dirt_grid_color.keys():
            print key_class            
            VizualMark().publish_marker(self.pose_by_class[key_class],sizes=[[grid_size, grid_size, 0.0]] * len(self.pose_by_class[key_class]),color=[dirt_grid_color[key_class]] *len(self.pose_by_class[key_class]))

    def plot(self, key='h'):
        self.classify_cell(key)
        classes = OrderedDict()
        for key_class in self.pose_by_class.keys():
            classes[key_class] = len(self.pose_by_class[key_class])
        x = []
        for k in range(class_num):
            x.append((k+1)*10)  
        y = []
        for values in classes.values():
            y.append(float(values) / float(cell_num))     
        Plot().plot_bar(x = x, y = y,x_labels= classes.keys(), width=4, x_axis = 'Classes', y_axis = 'Dirt num', labels_position='vertical')

    def write_dirt_in_time_to_file(self):
        file_dict = OrderedDict()
        headers = ['cell_dirt', 'time']
        time = [sum(cleaning_cycle.values()[:i]) for i in range(1,len(cleaning_cycle.values())+1)]
        if not os.path.isfile('/home/mtb/Documents/data/train/dirt_time_variation.csv'):
            for i in range(1, cell_num +1 ):
                file_dict[str(i)] = []
                for file in cleaning_cycle.keys():
                    file_dict[str(i)].append(csvconverter(csvread(grid_dirt_filename + file ), mode='int')['grid_dirt'][i-1])
                
                csvwriter('/home/mtb/Documents/data/train/dirt_time_variation.csv' , headers , [[file_dict[str(i)]], [time]])
            
        else:
            print 'Already written to file'

        return True

    def dirt_prob_plot(self, fig_type='line'):            
        # self.write_dirt_in_time_to_file()                      
        data = csvread('/home/mtb/Documents/data/train/dirt_time_variation.csv')
        data_cell_conv = convert_lists_to_int(data['cell_dirt'])
        time_cell_conv = convert_lists_to_int(data['time'])
        for i in range(len(time_cell_conv)):
                for j in range(len(data_cell_conv[str(i)][0])):
                    time_cell_conv[str(i)][0][j] = time_cell_conv[str(i)][0][j] *3600
        # print  time_cell_conv[str(0)][0]
        # print  data_cell_conv[str(149)][0]
        # fig = plt.figure()
        # plt.bar(time_cell_conv[str(0)][0], data_cell_conv[str(149)][0], width=4)
        # plt.savefig('/home/mtb/Documents/data/train/cell_dirt_in_time/cell')
        # plt.close(fig)
    

        for i in range(len(time_cell_conv)):
            Plot().plot_and_save(fig_type='bar',x =time_cell_conv[str(0)][0], y =data_cell_conv[str(i)][0],width=4,save_path='/home/mtb/Documents/data/train/cell_dirt_in_time/cell' + str(i+1))
            
    def visualize_freq(self,frq_file=[]):
        pose = Process().get_center()
        for files in frq_file:
            text_list = csvread(files)['grid_freq_val']
            for values in text_list.values():
                values = str(values)
            TextMarker().publish_marker(text_list, pose)

    def rviz_dirt(self, header='dirt_level'):
        self.get_reachable_centers()
        file_read = csvread(self.filename)     
        file_read_convert = [int(elem) for elem in file_read[header]]
        # print max(file_read_convert)
        dirt_level = []
        with open('/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/data/corners2.csv') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                index_list = []
                for elem in row[0].split(':'):
                    index_list.append(int(elem))
                for i in range(index_list[0], index_list[1]):
                    dirt_level.append(file_read_convert[i])
        # print dirt_level
        # print max(dirt_level),len(dirt_level)
        thresh = np.max(dirt_level[375:])
        color = [int(255 - float(dirt_level[i]) / float(thresh) * 255)
                for i in range(len(dirt_level))]    
        pose = self.reachable_centers
        # TextMarker().publish_marker(
        #     text_list=['FRIDGE','STOVE','TABLE','SOFA'],
        #      pose=[[1.14,7.8],[-0.7,7.7],[-0.1,5.5],[0.45,2.27]],angle=[0.0,0.0,0.0,0.0,1.57])

        VizualMark().publish_marker(pose, sizes=[[0.25,0.25,0.0]] * len(pose) , color= color, 
                        convert = True, texture='Red',action='add', id_list = range(len(pose)))
        # pose = [[-0.125, -2.875]]
        # VizualMark().publish_marker(pose,sizes=[[0.25,0.25,0.0]] * 1 ,color=['Green' ] * 1, action='delete',publish_num = 50, id_list=[0])
    def get_reachable_centers(self):
        self.get_center()
        pose = self.centers
        self.reachable_centers = []
        text= [str(i) for i in range(1296)]
        # po = [pose[i] for i in range(838,843)]
        # to = [pose[i] for i in range(865, 870)]
        # print po
        # VizualMark().publish_marker(pose=to, sizes=[[0.25,0.25,0.25]] * 5 , color=['Red'] * 5)
        # TextMarker().publish_marker(text,pose)
        with open('/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/data/corners2.csv') as csvfile:
            csvreader = csv.reader(csvfile)            
            for row in csvreader:              
                index_list = []
                for elem in row[0].split(':'):                    
                    index_list.append(int(elem))               
                for i in range(index_list[0], index_list[1]):
                    self.reachable_centers.append(self.centers[i])
        
        # print (self.reachable_centers)
        # VizualMark().publish_marker(pose=self.reachable_centers[:], sizes=[
        #     [0.25, 0.25, 0.25]] * len(self.reachable_centers[:]), color=['Red'] * len(self.reachable_centers[:]))
        # text = [str(i) for i in range(623)]
        # TextMarker().publish_marker(text,self.reachable_centers)
        # csvwriter(filepath='/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/data/pose.csv',headers=['pose'], rows = [self.reachable_centers])
        # print self.reachable_centers[140:145]
    def get_tracking_person_freq_in_cells(self, filenames= [], new_filename='new.csv'):
        dirt_level = []
        for files in filenames:
            # print files
            data = csvread(files)['grid_freq_val']
            for i in range(len(data)):
                try:
                    dirt_level[i] += float(data[i])
                except:
                    dirt_level.append(float(data[i]))
        csvwriter(new_filename, headers=['grid_freq_val'], rows=[dirt_level])
        # print (dirt_level[1218])
            
    def rviz_human(self, header='grid_freq_val'):
        self.get_reachable_centers()
        file_read = csvread(self.filename)
        file_read_convert = [float(elem) for elem in file_read[header]]
        # print max(file_read_convert)
        duration = []
        with open('/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/data/corners2.csv') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                index_list = []
                for elem in row[0].split(':'):
                    index_list.append(int(elem))
                for i in range(index_list[0], index_list[1]):
                    duration.append(file_read_convert[i])
        max_time = 500.0
        # print duration
        human_list =[]
        for i in range(len(duration)):
            time_in_second= duration[i] * 0.025
          
            if time_in_second >=0.025:
                a = 255.0 - (time_in_second / max_time) * 255.0
                b = int (a)
               
                if b <=0:
                    human_list.append(0)
                else:
                    human_list.append(b)
            else:
                human_list.append(255)
                
        
        pose = self.reachable_centers
        # print human_list
        VizualMark().publish_marker(
            pose, sizes=[[0.25, 0.25, 0.0]] * len(pose), color=human_list, convert=True, texture='Red')
       
    def plot_dirt_human_duration_param(self, dirt_files =[], tracking_files = []):
        print dirt_files
        print tracking_files
        x_values = []
        y_values = []
        for i in range(len(dirt_files)):           
            dirt_read = csvread(dirt_files[i])['dirt_level']
            track_read = csvread(tracking_files[i])['grid_freq_val']
            # print track_read
            for j in range(len(dirt_read)):
                y_values.append(int(dirt_read[j]))
                x_values.append(float(track_read[j]))
        # print y_values
        # print x_values
        # import matplotlib.pyplot as plt
        # plt.plot(x_values, y_values, 'bo')
        # # plt.axis([0, 10, 0, max(y_values)]) 
        # plt.show()

        csvwriter(filepath='/media/mtb/Data Disk/data/extraction/human dirt/human_and_dirt_high.csv',
                  headers=['human_presence', 'dirt_level'], rows=[x_values, y_values])

    def extract_dirt_air_influence(self, key= 'h'):
        positions = range(838, 843) + range(865, 870) 
        cells = OrderedDict() 
        for j in positions:
            cells[j] = []
        if key == 'h':
            path = '/media/mtb/Data Disk/data/extraction/high level sensor/'
        elif key =='l':
            path = '/media/mtb/Data Disk/data/extraction/low level sensor/'
        for files in os.listdir(path):
            data = csvread(path + files)['dirt_level']
            for keys in cells:
                cells[keys].append(data[int(keys)])
            # print cells      
        csvwriter('/home/mtb/Desktop/air_influence_low.csv',
                  headers=cells.keys(), rows=cells.values())
    
    def get_air_influence_data(self,cells=[],key ='h'):
        # data_dict = OrderedDict()
        dataframe = DataFrame()
        dataframe_list = []
        if key == 'h':
            path = '/media/mtb/Data Disk/data/extraction/air influence/air_influence_high.csv'
        elif key == 'l':
            path = '/media/mtb/Data Disk/data/extraction/air influence/air_influence_low.csv'
        
        # x = [0,22,5,62,51,5,20,7,62,8,16,8,18,7,18,25,7,88,7,16,24,
        #          290,23,48,96,99,69,27,23,119,22,24,
        #          50,45,27,46,49,72,23,25,27,95,31,18]
        duration = [290,23,48,96,99,69,27,23,119,22,24,
                 50,45,27,46,49,72,23,25,27,95,31,18]
        duration = [sum(duration[0:j]) for j in range(len(duration) + 1)]
        for cell in cells:
            y = []
            data = csvread(path)[cell][21:]
            for i in range(len(data)):
                data[i] = int(data[i])
                y.append(int(data[i]))
            z  = [sum(y[0:j]) for j in range(len(y)+1)]
            # data_dict[cell] = z
            dataframe = DataFrame(Series(data=z,index=duration), columns=[cell])
            dataframe_list.append(dataframe)
        data = pd.concat(dataframe_list, join='outer', axis=1)
        data.to_csv('/media/mtb/Data Disk/data/extraction/air influence/high_level_duration.csv')
            # w = [sum(x[0:j]) for j in range(len(x))]
            # Plot().plot_and_save(fig_type='bar', x=w, y=z,color = 'green',
            #                      save_path='/home/mtb/12-25/air influence/low/time development/' + cell + '.png')
            # Plot().plot_and_save(fig_type='points',marker ='ro', x=x, y=y,
            #                      save_path='/home/mtb/12-25/air influence/high/time development/' + cell + '.png')

    def plot_air_influence(self,key='h'):
        if key =='h':
           data = pd.read_csv('/media/mtb/Data Disk/data/extraction/air influence/high_level_duration.csv', index_col=[0])
        else:
            data = pd.read_csv('/media/mtb/Data Disk/data/extraction/air influence/low_level_duration.csv')
        
        cells =  data.columns.values
        duration = data.index.values
        for cell in cells:
            Plot().plot_and_save(fig_type='points', marker='bo', x=list(duration), y=list(data[cell]), save_path='/home/mtb/01-04/low/' + cell +'.png')
    
    def reorder_human_csv_file(self):
        path = '/home/mtb/01-04/id/'
        new_path = '/media/mtb/Data Disk/data/extraction/human dirt/'
        tracking_dict = {
        '12_5.csv': ['12-4_noon.csv','12-5_noon.csv' ],
        '12_7.csv': ['12-6_morning.csv' ,'12-6_noon.csv'],
        '12_9.csv': ['12-7_noon.csv',  '12-8_morning.csv', '12-8_noon.csv'],
        '12_12.csv': ['12-11_noon.csv',  '12-12_morning.csv'],
        '12_13.csv': ['12-12_noon.csv',  '12-13_morning.csv'],
        '12_14.csv': ['12-13_noon.csv',  '12-14_morning.csv'],
        '12_15.csv': ['12-14_noon.csv',  '12-15_morning.csv'],
        '12_19.csv': ['12-15_noon.csv',  '12-16_noon.csv', '12-17_noon.csv', '12-18_morning.csv', '12-18_noon.csv', '12-19_morning.csv'],
        '12_20.csv': ['12-19_noon.csv', '12-20_noon.csv'],
        '12_21.csv': ['12-21_morning.csv', '12-21_noon.csv'],        
        }
        for dates in tracking_dict:
            filenames = [path + tracking_dict[dates][i] for i in range(len(tracking_dict[dates]))]
            self.get_tracking_person_freq_in_cells(filenames=filenames, new_filename=new_path + dates)
 
    def find_cent(self):
        self.get_center()
        self.reachable_centers = []
        self.t = []
        with open('/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/data/corners.csv') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                index_list = []
                for elem in row[0].split(':'):
                    index_list.append(int(elem))
                for i in range(index_list[0], index_list[1]):
                    self.reachable_centers.append(self.centers[i])
        
        with open('/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/data/corners2.csv') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                index_list = []
                for elem in row[0].split(':'):
                    index_list.append(int(elem))
                for i in range(index_list[0], index_list[1]):
                    self.t.append(self.centers[i])
        print len(self.reachable_centers), len(self.t)
        for elem in self.t:
            if elem not in self.reachable_centers:
                print elem

if __name__ == '__main__':    
    # Process().get_center()
    # pose = []
    # a = Process().get_rectangle_corners()
    # for elem in a:
    #     pose.append(a[elem][0])
    # np.save('/home/mtb/array.npy', np.array(pose))
        
    rospy.init_node('path')
    path = '/home/mtb/Documents/data/dirt_extraction_2/data/high/dirt_value/'
    for files in sorted(os.listdir(path))[15:]:
        print files
        Process(filename=path + files,files=files).rviz_dirt()
        # break;
        # break;
    # Process(filename='/home/mtb/Documents/data/dirt_extraction_2/data/low/dirt_value/test2017-02-01_low.csv').get_total_dirt_num()
    # Process().get_reachable_centers()
#     pose = [[-0.625, 6.625],
# [-0.375, 6.625],
# [-0.125, 6.625]]
#     VizualMark().publish_marker(pose,sizes=[[0.25,0.25,0.0]] * 3 ,color=['Green' ] * 3, action='add',publish_num = 50)
    # a = Process().get_center()
    # np.save('/home/mtb/rulo_ws/src/rulo_pkg/rulo_ml/train_data/positions.npy',np.array(list(a)))
    # Process().reorder_human_csv_file()
    # print Process(filename='/media/mtb/Data Disk/data/dirt/test2017-10-06_11h_15.csv').get_dirt_pose_and_dirt_level()[1]['h']
    
    # Process().get_reachable_centers()
    # Process('/media/mtb/Data Disk/data/extraction/human dirt/12_15.csv').rviz_human()
    # Process().reorder_human_csv_file()
    # Process().get_air_influence_data(cells=['838','839','840','841','842','865','866','867','868','869'],
                                #  key='h')
    # Process().plot_air_influence(key='l')
    # path = '/home/mtb/DIRT/'
    # filenames = [path + files for files in ['12-14_morning.csv',
    #                                         '12-14_noon.csv']]
    # print filenames[0]
    # Process().get_tracking_person_freq_in_cells(filenames= filenames, new_filename= path + '12_15.csv')

    # for files in os.listdir(path):
    #     print files
    # csvread('/home/mtb/dirt_num/12-5_noon.csv')
    # # path = '/media/mtb/Data Disk/data/tracking_csv/'
    # path = '/media/mtb/Data Disk/data/extraction/high level sensor/'
    # # # path = '/media/mtb/Data Disk/data/extraction/low level sensor/'
    # # # Process(filename=path + 'test2017-12-21_14h38.csv').rviz_dirt()

    # dirt_files = [path + files 
    #             for files in ['test2017-12-07_10h15.csv', 'test2017-12-09_11h10.csv', 
    #                           'test2017-12-12_11h05.csv', 'test2017-12-13_10h18.csv',
    #                           'test2017-12-14_11h10.csv', 'test2017-12-15_14h30.csv',
    #                           'test2017-12-19_13h40.csv', 'test2017-12-20_20h30.csv', 'test2017-12-21_14h38.csv'
    #                                ]]
    
    # tracking_files = ['/media/mtb/Data Disk/data/extraction/human dirt/' +
    #                   files for files in ['12_7.csv', '12_9.csv', '12_12.csv', '12_13.csv', '12_14.csv', '12_15.csv',
    #                                        '12_19.csv','12_20.csv','12_21.csv']]

    # Process().plot_dirt_human_duration_param(dirt_files, tracking_files)
    # for files in ['test2017-12-19_13h40.csv', 'test2017-12-20_20h30.csv', 'test2017-12-21_14h38.csv']:
    #     Process(filename='/media/mtb/Data Disk/data/dirt/' +files, files=files).get_grid_dirt('l')
    # for files in  os.listdir(path):
    #         print files
    #         Process(files).get_grid_dirt(key='l')
    # Process(id_num=43).get_room_cleaned_centers(key='l')
    # print csvread(cleaned_cell_filename)['pose'][0]
    # Process().visualize_cleaned_dirt()
    # date_list=['12-4', '12-7', '12-9']
    # key ='l'
    # files = [report_filename+ date_list[i] + '_'+key+'.csv' for i in range(len(date_list))]
    # print files
    # y = []
    # for file in files:
    #     data = csvread(file)['dirt_num']
    #     value = []
    #     for dirt in data:
    #         value.append(int(dirt))
    #     y.append(sum(value))
    
    # Plot().plot_bar(x=[3,6,9], y=y, x_labels= date_list, width=1)
        # ['test2017-12-05_12h25.csv'] = ['12-4_noon.csv']
        # ['test2017-12-07_10h15.csv'] = ['12-5_noon.csv', '12-6_morning.csv', '12-6_noon.csv']
        # ['test2017-12-09_11h10.csv'] = ['12-7_noon.csv','12-8_morning.csv', '12-8_noon.csv', '12-9_morning.csv']
        # ['test2017-12-12_11h05.csv'] = ['12-9_noon.csv','12-11_noon.csv','12-12_morning.csv']
        # ['test2017-12-13_10h18.csv'] = ['12-12_noon.csv', '12-13_morning.csv']
        # ['test2017-12-14_11h10.csv'] = ['12-13_noon.csv', '12-14_morning.csv']
        # ['test2017-12-15_14h30.csv'] = ['12-14_noon.csv', '12-15_morning.csv']
        # ['test2017-12-19_13h40.csv'] = ['12-15_noon.csv', '12-16_noon.csv', '12-17_noon.csv','12-18_morning.csv', '12-18_noon.csv', '12-19_morning.csv']
        # ['test2017-12-20_20h30.csv'] = ['12-19_noon.csv', '12-20_noon.csv']
        # ['test2017-12-21_14h38.csv'] = ['12-21_morning.csv','12-21_noon.csv']
        # a =  np.load('/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/data/all_cells_pose.npy')
        # print a.shape
