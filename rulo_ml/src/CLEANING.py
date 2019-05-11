#!/usr/bin/env	python
import os
import rospy
import csv
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from collections import deque, OrderedDict
from rulo_utils.csvreader import csvread
from rulo_utils.csvwriter import csvwriter

corners = [[-2.25, -4.0], [-2.25, 8.0], [4.5, 8.0], [4.5, -4.0]]
grid_size = 0.25
from rulo_base.markers import VizualMark, TextMarker

class Cleaning:
    
    def __init__(self):    
        self.centers = deque()
            
    def get_center(self):
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
    
    def find_centers(self):
        self.get_center()
        self.centers_rl_1 = []
        self.centers_rl_2 = []
        with open('/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/data/corners.csv') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                index_list = []
                for elem in row[0].split(':'):
                    index_list.append(int(elem))
                for i in range(index_list[0], index_list[1]):
                    self.centers_rl_1.append(self.centers[i])
        
        with open('/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/data/corners2.csv') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                index_list = []
                for elem in row[0].split(':'):
                    index_list.append(int(elem))
                for i in range(index_list[0], index_list[1]):
                    self.centers_rl_2.append(self.centers[i])
        indice =[]
        for elem in self.centers_rl_2:
            indice.append(list(self.centers).index(elem))
        # np.save('/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/data/indices.npy', np.array(indice))
        # csvwriter('/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/data/pose2.csv', 
        #             ['pose'], [self.centers_rl_2])
        # np.save('/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/data/pose.npy', np.array(self.centers_rl_2))
        # self.ignore_id1 = []
        # self.ignore_id2 = []
        # for elem in self.centers_rl_1:
        #     if elem not in self.centers_rl_2:
        #         self.ignore_id1.append(self.centers_rl_1.index(elem))
                
        # for elem in self.centers_rl_2:
        #     if elem not in self.centers_rl_1:
        #         print elem
        #         self.ignore_id2.append(self.centers_rl_2.index(elem))
                
        # print self.ignore_id1
        # print self.ignore_id2
        # for elem in self.ignore_id2:
        #     print self.centers[elem]
        # rospy.init_node('rnn')
        # VizualMark().publish_marker(self.centers_rl_2,sizes=[[0.25,0.25,0.25]] * len(self.centers_rl_2), 
        #                             color=['Red'] * len(self.centers_rl_2))
    def get_data(self):
        folders_list = ['0-30', '30-43', '43-50', '50-75', '75-90', '90-100', '100-130', '130-150', '150-180', '180-200',
                        '200-240', '240-250', '250-290', '290-300', '300-350', '350-400', '400-450', '450-465', '465-475', '475-500',
                        '500-550', '550-575', '575-600', '600-630', '630-649']
        folders_exc = ['650', '651', '652', '653']
        path = '/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/DATA/'
        # res = OrderedDict()
        for folders in folders_list:
            print folders              
            indice_range = map(int,folders.split('-'))
            # index_file = csvread(path + folders + '/index.csv')
            origin_file = csvread(path + folders + '/origin.csv')
            time_file = csvread(path + folders + '/time.csv')
            res = OrderedDict()
            for i in range(indice_range[0],indice_range[1]):
            # for i in range(649,654):
                res[str(i)] = OrderedDict()
                for m in range(len(origin_file['pose'])):
                    if origin_file['pose'][m] == str(i):
                        if origin_file['goal_state'][m] == '3':
                            res[str(i)]['1000'] = float(origin_file['finish_time'][m]) - float(origin_file['start'][m])
                        else:
                            res[str(i)]['1000'] = 200.0
                for j in range(i+1,650):
                    for m in range(len(time_file['pose'])):
                        pose =map(int,time_file['pose'][m][1:-1].split(','))
                        if [i,j] == pose:
                            if time_file['goal_state'][m] == '3':
                                res[str(i)][str(j)] = float(time_file['finish_time'][m]) - float(time_file['start'][m])
                            else:
                                res[str(i)][str(j)] = 200.0     
                for j in range(650,654):
                    # new_index_file = csvread(path + str(j)+ '/index.csv')
                    new_origin_file = csvread(path + str(j)+ '/origin.csv')
                    new_time_file = csvread(path + str(j)+ '/time.csv')
                    for m in range(len(new_time_file['pose'])):
                        pose = map(int, new_time_file['pose'][m][1:-1].split(','))
                        if [j, i] == pose:
                            if new_time_file['goal_state'][m] == '3':
                                res[str(i)][str(j)] = float(
                                    new_time_file['finish_time'][m]) - float(new_time_file['start'][m])
                            else:
                                res[str(i)][str(j)] = 200.0
                          
                dataframe = DataFrame(data=res[str(i)].values(), index=res[str(i)].keys() ,columns=[str(i)])  
                dataframe.to_csv(path + 'csv/' +str(i) +'.csv')
                del res[str(i)]
                
    def reorder(self):
        pose_filename = '/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/data/pose.csv'
        self.get_center()
        data = csvread(pose_filename)
        self.position = []
        for str_pose in data['pose']:
            pose = []
            for elem in str_pose[1:-1].split(','):
                pose.append(float(elem))
            self.position.append(pose)
        self.index = OrderedDict()
        for elem in self.position:
            self.index[self.position.index(elem)] = list(self.centers).index(elem)
        a = OrderedDict(sorted((value, key) for (key, value) in self.index.items()))
        path = '/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/DATA/csv/'
        train_1 = [314, 369, 492, 505, 506, 507, 508, 509, 521, 522, 523, 524, 525, 537, 538,
                   539, 540, 541, 552, 553, 554, 555, 556, 642, 643, 644, 645, 646, 647, 648, 649]
        # # train_2 = [434, 449, 455, 471]
        result = []
        columns = []
        for i in range(654):
            res = pd.read_csv(path + str(i) +'.csv', index_col=[0])
            result.append(res)
            columns.append(str(i))
        dataframe = pd.concat(result, axis=1)        
        dataframe = dataframe.reindex(index =range(0,654) + [1000])
        dataframe.iloc[0,1:] = np.array([200.0]*len(range(653)))
        
        for j in range(1,654):
            for t in range(1,j+1):
                if t !=j:
                    dataframe[str(j)][t] = dataframe[str(t)][j]
        for elem in train_1:
            dataframe = dataframe.drop([elem])
            dataframe = dataframe.drop([str(elem)],axis=1)
        # print (dataframe.columns.values.shape)
        b = OrderedDict()
        for cells_num in dataframe.columns.values:
            val = a.values().index(int(cells_num))
            b[cells_num] =  a.keys()[val]
        dataframe.rename(columns=b, inplace=True)
        c = dict(zip(map(int, b.keys()),b.values()))
        c[1000] = 10000
        dataframe.rename(index=c, inplace=True)
        d = sorted(dataframe.columns.values)
        e = dict(zip(d, range(623) ))
        dataframe.rename(columns=e, inplace=True)
       
        f = dict(zip(d, range(623)))
        f[10000] = 10000
        # print f
        #  dataframe = dataframe.sortlevel(0)
        dataframe.rename(index=f, inplace=True)
        dataframe = dataframe.sortlevel(0)
        dataframe = dataframe[e.values()]

        dataframe.to_csv('/home/mtb/rulo_ws/src/rulo_pkg/rulo_ml/data/cells_time.csv')
        # print (dataframe.columns.values.shape)  
    
    def get_dirt(self):
        dirt = csvread(
            '/home/mtb/Documents/data/dirt_extraction_2/data/high/test2017-02-19_estimation.csv')
        indice = np.load('/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/data/indices.npy')
        dirt_level = map(float,dirt['dirt_level'])
        dirt =[]
        for elem in indice:
            dirt.append(dirt_level[elem])
        np.save(
            '/home/mtb/rulo_ws/src/rulo_pkg/rulo_ml/dqn/data/dirt_high_19.npy', np.array(dirt))
if __name__ == '__main__': 
    Cleaning().get_dirt()
    # Cleaning().find_centers()
    # Cleaning().reorder()
    # Cleaning().get_data()
    # a =[1,2,3,4,5,6]
   
