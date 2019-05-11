#!/usr/bin/env	python
import rospy
import os
import numpy as np
from collections import deque, OrderedDict
from rulo_utils.csvreader import csvread
from rulo_utils.csvcreater import csvcreater
from rulo_utils.csvwriter import csvwriter
from rulo_utils.csvconverter import csvconverter
from math import copysign, fabs
from rulo_base.markers import VizualMark, TextMarker
from rulo_base.path_creater import Path
from rulo_utils.graph_plot import Plot
from matplotlib import pyplot as plt
import time
import datetime
from math import pow

path = '/home/mtb/Documents/data/'
filename = ['test2017-10-04_11h_45.csv', 'test2017-10-05_16h.csv', 'test2017-10-06_11h_15.csv', 'test2017-10-06_21h.csv',
            'test2017-10-07_13h.csv', 'test2017-10-07_18h25.csv', 'test2017-10-10_08h20.csv', 'test2017-10-10_19h.csv',
            'test2017-10-11_12h15.csv', 'test2017-10-11_19h.csv', 'test2017-10-12_11h20.csv', 'test2017-10-13_14h.csv',
            'test2017-10-13_21h.csv', 'test2017-10-16_10h45.csv', 'test2017-10-16_18h30.csv', 'test2017-10-17_10h30.csv',
            'test2017-10-17_18h30.csv', 'test2017-10-18_12h20.csv', 'test2017-10-18_19h10.csv', 'test2017-10-19_13h10.csv',
            'test2017-10-20_12h30.csv', 'test2017-10-20_19h30.csv', 'test2017-10-24_11h30.csv', 'test2017-10-24_18h40.csv',
            'test2017-10-25_11h00.csv', 'test2017-10-26_10h45.csv','test2017-11-07_12h35.csv'
            ]

corners = [[-1.0, 2.0], [2.0, 2.0], [2.0, 1.0], [4.0, 1.0], [4.0,-2.0],[-2.0,-2.0],[-2.0, -1.0],[-1.0,-1.0]]
grid_size = 0.25
dirt_size = 0.02
max_dirt = 100
color = {'low': 'Blue', 'medium': 'Yellow', 'high': 'Red'}

grid_dirt_filename = '/home/mtb/Documents/data/' + 'train/'
cell_num = 1296
class_num = 100


class Process:
    def __init__(self, id_num=None):
        self.pose = deque()
        self.dirt_h_level,  self.dirt_l_level, self.wall_time = deque(), deque(), deque()
        self.laser_ranges = deque()
        self.corners = OrderedDict()
        self.centers = deque()

        self.grid_point = OrderedDict()
        self.labels = deque()
        self.input = deque()
        if id_num:
            self.file_id = id_num
            print self.file_id
            self.data = csvread(path + filename[self.file_id])
     
    
    def split_in_square(self):
        y_list, x_list = [], []
        for corner in corners:
                if corner[1] not in y_list:
                    y_list.append(corner[1])
                if corner[0] not in x_list:
                    x_list.append(corner[0])
        y_list_arranged_reverse= sorted(y_list, reverse=True)
        # x_list_arranged = sorted(x_list)
        # points = OrderedDict()
        # for y_line in y_list_arranged_reverse:
        #     points[y_line] = []
        #     for corner in corners:
        #         if corner[1] == y_line:
        #             points[y_line].append(corner)
        # y_line_pair =[]
        # for key, value in points.items():
        #     for i in range(len(value) -1):
        #         if corners.index(value[i+1]) == corners.index(value[i]) + 1:
        #             y_line_pair.append([value[i], value[i+1]])
        
        # print y_line_pair
        # squares = []
        # for i in range(len(y_line_pair)-1):
        #     p1 = y_line_pair[i][0]
        #     p2 = y_line_pair[i][1]
        #     print [p1,p2]
        #     for j in range(len(y_line_pair[i+1])):
        #         print y_line_pair[i+1][j]
        #         print corners.index(y_line_pair[i+1][j]), corners.index(p2) + 1
         
        #         if corners.index(y_line_pair[i+1][j]) ==corners.index(p2) + 1:
        #              print 'true'
        #         if corners.index(y_line_pair[i+1][j]) == corners.index(p2) + 1:
        #             p3 =  y_line_pair[i+1][j]
        #             p4 =  [y_line_pair[i][0][0],y_line_pair[i+1][j][1]]

        #             squares.append([p1,p2,p3,p4])
        # print squares
        y_line = False
        x_line = True
        square = []
        
        corner_list = corners
        start = corner_list[0]
        self.line_position()
        start_horizontal= False
        start_vertical = True
        num_square = len(y_list_arranged_reverse) - 1 
        square = self.trace_lines(y_list_arranged_reverse, corner_list, num_square)
        print square
    
    def trace_lines(self, y,corner_list, num):
        val_x = []    
        pairs =[]   
        for x in self.vertical.keys():
            val_x.append(float(x))       
        for y_val in y[1:num]:           
            y_list = []
            for x in sorted(val_x):
                y_list.append([x, y_val])            
            
            xmin, xmax = corner_list[0][0], corner_list[1][0]

            for y_points in y_list:
                if y_points[0] ==xmin:
                    p4 = y_points
                if y_points[0] ==xmax:
                    p3 = y_points
            
            p1, p2 = corner_list[0], corner_list[1]
            pairs.append((p1, p2, p3, p4))
            # print pairs[-1]
            corner_list = self.rearrange(corner_list, y_val, p3, p4)
            # print corner_list
        return pairs
    def rearrange(self, corn_list, y_line, p3,p4):
        xlimitmax, xlimitmin = p3[0], p4[0]
        for i in range(len(self.horizontal[str(y_line)])):
            if xlimitmin > min(self.horizontal[str(y_line)][i]):
                xlimitmin = min(self.horizontal[str(y_line)][i])
            if xlimitmax < max(self.horizontal[str(y_line)][i]):
                xlimitmax = max(self.horizontal[str(y_line)][i])

        p1, p2 = [xlimitmin, y_line], [xlimitmax, y_line]

        corn_list =[p1,p2]
        return corn_list 



            # point_list = []
            # x_min, x_max = 
            # for point_y in y_list:
            #     if self.vertical[str(point_y[0])]:
                                   





        
        # for i in range(1):
        #     p1, p2, p3 = [corner_list[0][0], y_list_arranged_reverse[i]], [corner_list[1][0], y_list_arranged_reverse[i]],\
        #                          [corner_list[2][0], y_list_arranged_reverse[i+1]]
        #     p4 =[p1[0], p3[1]]
        #     print p4

        # completed = False
        # while not completed:
            
        # for i in range(len(corners)):
        #     p1 = corner_list[0]
        #     p2 = corner_list[1]
        #     p3 = corner_list[2]
        #     if start_vertical:
        #         p4 = [p1[0], p3[1]]
        #         xbelong = False
        #         if str(p4[0]) in self.vertical.keys():      
        #             print p4       
        #             for elem in self.vertical[str(p4[0])]:
        #                 if min(elem)<p4[1]<max(elem):                            
        #                     square.append((p1,p2,p3,p4))
        #         start_vertical = False
        #         start_horizontal = True                            

        #     elif start_horizontal:
        #         p4 = [p3[0], p1[1]]
        #         if str(p4[1]) in self.horizontal.keys():
                    
        #             for elem in self.horizontal[str(p4[1])]:
        #                 if min(elem) <= p4[0] <= max(elem):
        #                     square.append((p1, p2, p3, p4))
        #         start_vertical = True
        #         start_horizontal = False
                           
        #     # print square,start_vertical
        #     corner_list = self.rotate(corner_list)
        
        # print square
        # print self.horizontal
        # for i in range(1) :
           
                        
            # p1 = corner_list[0]
            # p2 = corner_list[1]
            # if y_line:
            #     p3 = corner_list[2]
            #     p4 = [p3[0], p1[1]]
            #     print p1,p2,p3,p4
            #     y = p4[1]
            #     x = p4[0]
            #     for corner in corner_list[3:]:
            #         if corner[1] == y:
            #             x_points = corner[0]
                
                # for corner in corner_list[3:]:
                #     if corner[0] < p4[0] < corner_list[corner_list.index(corner)-1][0]\
                #      and corner[1] == p4[1] and corner_list[corner_list.index(corner) - 1][1]:
                #         square.append((p1,p2,p3,p4))
                #         y_line = True
                #         x_line = False   
                #         break         
            # if x_line:
            #     p3 = corner_list[2]
            #     p4 = [p1[0], p3[1]]
            #     y = p4[1]
            #     x = p4[0]  

            #     print p1,p2,p3,p4           
            #     for corner in corner_list[3:]:
            #         if corner[0]== x:
            #             y_points = corner[1]
               
            #     ymin = min([y_points, p1[1]])
            #     ymax = max([y_points, p1[1]])
                
            #     if ymin < y < ymax:
            #             square.append((p1,p2,p3,p4))
            #             x_line = False
            #             y_line = True
                       
            # corner_list = self.rotate(corner_list, indice=1)
    
    def line_position(self):
        self.vertical = OrderedDict()
        self.horizontal = OrderedDict()
        vertical = False
        horizontal = True
        for i in range(1,len(corners)):                  
            if vertical:
                x = corners[i][0]
                if str(y) not in self.vertical.keys():
                    self.vertical[str(x)] = [[corners[i-1][1], corners[i][1]]]
                else:
                  self.vertical[str(x)].append([corners[i-1][1], corners[i][1]])
                    
                horizontal = True
                vertical = False
                
            elif horizontal:
                y = corners[i][1]
                if str(y) not in self.horizontal.keys():
                    self.horizontal[str(y)] = [[corners[i-1][0], corners[i][0]]]
                else:
                    
                    self.horizontal[str(y)].append([corners[i-1][0], corners[i][0]])
               
                horizontal = False
                vertical = True
        
        if vertical:
                x = corners[0][0]
                if str(y) not in self.vertical.keys():
                    self.vertical[str(x)] = [[corners[-1][1], corners[0][1]]]
                else:
                  self.vertical[str(x)].append([corners[-1][1], corners[0][1]])
                    
                horizontal = True
                vertical = False
                
        elif horizontal:
                print horizontal
                y = corners[0][1]
                if str(y) not in self.horizontal.keys():
                    self.horizontal[str(y)] = [[corners[-1][0], corners[0][0]]]
                else:
                    self.horizontal[str(y)].append([corners[-1][0], corners[0][0]])
        # print self.vertical
        # print self.horizontal
            
    
    def rotate(self,list_,indice=1):
        return (list_[indice:] + list_[:indice])
        
                        
                    
                
            
                    
                   
            

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

    
if __name__ == '__main__':
    rospy.init_node('rnn')
    output = Process().split_in_square()
