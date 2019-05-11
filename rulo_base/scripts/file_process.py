#!/usr/bin/env	python
import	rospy
import numpy as np
from collections import deque, OrderedDict
from rulo_utils.csvreader import csvread
from rulo_utils.csvcreater import csvcreater
from rulo_utils.csvwriter import csvwriter
from math import copysign, fabs
from rulo_base.markers import VizualMark, TextMarker
from rulo_base.path_creater import Path
# from rulo_utils.graph_plot import plot
from matplotlib import pyplot as plt
import time
#Change filename depending on the date

path = '/media/mtb/Data Disk/data/'
# path='/home/mtb/Documents/data/'
filename = ['test2017-10-04_11h_45.csv', 'test2017-10-05_16h.csv', 'test2017-10-06_11h_15.csv', 'test2017-10-06_21h.csv',
            'test2017-10-07_13h.csv', 'test2017-10-07_18h25.csv', 'test2017-10-10_08h20.csv', 'test2017-10-10_19h.csv',
            'test2017-10-11_12h15.csv', 'test2017-10-11_19h.csv', 'test2017-10-12_11h20.csv', 'test2017-10-13_14h.csv',
            'test2017-10-13_21h.csv', 'test2017-10-16_10h45.csv', 'test2017-10-16_18h30.csv', 'test2017-10-17_10h30.csv',
            'test2017-10-17_18h30.csv', 'test2017-10-18_12h20.csv', 'test2017-10-18_19h10.csv', 'test2017-10-19_13h10.csv',
            'test2017-10-20_12h30.csv', 'test2017-10-20_19h30.csv', 'test2017-10-24_11h30.csv', 'test2017-10-24_18h40.csv',
            'test2017-10-25_11h00.csv', 'test2017-10-26_10h45.csv', 'test2017-11-07_12h35.csv',  'test2017-11-08_11h15.csv',
            'test2017-11-10_11h40.csv', 'test2017-11-14_11h40.csv', 'test2017-11-16_12h00.csv', 'test2017-11-17_12h40.csv',
            'test2017-11-18_15h00.csv', 'test2017-11-21_11h45.csv', 'test2017-11-22_14h00.csv', 'test2017-11-23_13h00.csv'

            ]


# csvcreater(name, fieldnames= fieldnames)
# name = '/home/mtb/Documents/data/test2017-10-17_18h30_v2.csv'
# corners=[[-5,5],[2,5],[2,1],[5,1],[-5,-7],[5,-7]]
# corners=[[2,4],[4,4],[4,2],[2,2]]

# corners = [[0.0,0.0],[2.0,2.0],[1.0,1.0],[2.0,0.0],[0.0,1.0],[1.0,2.0]]

# corners = [[1.0,2.0],[2.0,2.0],[2.0,1.0],[3.0,1.0],[1.0,1.0],[0.0,1.0],[0.0,0.0],[2.0,0.0],[1.0,0.0],[3.0,0.0],[2.0,-1.0],[1.0,-1.0],
#                     [3.0,2.0],[4.0,2.0],[4.0,0.0]]

# corners = [[2.0,-3.0],[0.0,-3.0],[0.0,-1.0],[-2.0,-1.0],[-2.0,1.0],[0.86,1.0],[0.86,2.7],[-0.37,2.7],
#                      [-0.37, 4.1], [-0.82, 4.1], [-0.82,3.65],[-1.88, 3.65],[-1.88,4.3],[-2.2, 4.3],[-2.2, 5.6],
#                      [-2.2,5.6],[-2.0,5.6],[-2.0,6.5],[-1.7,6.5],[-1.7,7.3],[0.75,7.3],[0.75,7.1],[1.37,7.1],
#                      [1.37,7.9],[2.14,7.9],[2.14,6.3],[1.6,6.3],[1.6,4.13],[2.0,4.13],[2.0,4.86],[4.3,4.86],
#                      [4.3,4.86],[4.3,4.32],[3.89,4.32],[3.89,3.64],[3.89,3.64],[4.3,3.64],[4.3,0.34],[2.0,0.34]
# ]

# corners=[[-2.0, 0.0],[-2.0,2.0],[2.0,2.0],[2.0,0.0]]
# corners=[[0.0, 0.0],[0.0,2.0],[2.0,2.0],[2.0,0.0]]

corners=[[-2.25,-4.0],[-2.25,8.0],[4.5,8.0],[4.5,-4.0]]
grid_size = 0.25
dirt_size = 0.02
dirt_level_threshold = 5000 
color = {'low': 'Blue', 'medium': 'Yellow', 'high': 'Red'}
# file_id = 17

class Process:
    def __init__(self, id_num= None):        
        self.x_pose, self.y_pose,self.theta = deque(), deque(), deque()
        self.pose=deque()
        self.dirt_h_level,  self.dirt_l_level= deque(), deque()
        self.laser_ranges = deque()
        self.center_list = deque()
        self.corners = OrderedDict()
        self.centers = deque()
        self.pose_dirt , self.dirt, self.dirt_indice, self.dirt_num=OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()
        self.grid_point = OrderedDict()     
        self.labels=deque()  
        self.input = deque()
        if  id_num:
            self.file_id = id_num   
            print self.file_id
            self.data = csvread(path + filename[self.file_id])
            print path + filename[self.file_id]
    
    def empty_list(self):
        for elem in[self.x_pose, self.y_pose,self.theta,self.pose,self.dirt_h_level,  self.dirt_l_level,  self.laser_ranges ,self.center_list ,
                     self.centers, self.labels, self.input]:
            del elem

    def get_center(self):
        '''
        Returns a list of centers position list
        ---
        self.center_list = [[0.5,0.5],[1.0,1.0],...]
        '''        
        
        x_start, y_start = corners[0][0], corners[0][1]
        self.center_list.append([x_start + grid_size/2.0, y_start + grid_size/2])
        center_start =self.center_list[0]
        list_x, list_y = deque(), deque()
        for [x,y] in corners:
            list_x.append(x)
            list_y.append(y)
        increment = 0.0
        while(self.center_list[-1][1]< max(list_y)):
             while(self.center_list[-1][0] < max(list_x)):                
                 val = [self.center_list[-1][0] + grid_size , self.center_list[-1][1]]
                 if (val[0] < max(list_x)):
                     self.center_list.append(val)                     
                 else:
                    break           
             if self.center_list[-1][1] + grid_size < max(list_y):
                 self.center_list.append([self.center_list[0][0] , self.center_list[-1][1] +grid_size])
             else:
                break       
        return self.center_list
    
    def get_rectangle_corners(self):
        '''
        Returns a list of centers coordinates and a dictionnary of corners coordinates 
        ---
        self.center = [[0.5,0.5],[1.0,1.0],...]     
        self.corners = {'[0.5,0.5]':[[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]],...}
        '''
        
        for center in self.get_center():
            self.centers.append(center)            
            self.corners[str(center)]=[[center[0]-grid_size/2, center[1] -grid_size/2],
                                                               [center[0] + grid_size/2, center[1] - grid_size/2],
                                                               [center[0] + grid_size/2, center[1] + grid_size/2],
                                                               [center[0]-grid_size/2, center[1] + grid_size/2]]
        return self.centers , self.corners

    def get_pose_and_dirt_level(self):
        '''
        Returns a list of pose for each axis, dirt level for each sensor
        ---
        self.x_pose = [0.25, 0.5,...]
        self.y_pose = [0.25, 0.5,...]
        self.pose = [[0.25, 0.25],...]
        self.dirt_h_level = [280, 282,...]
        self.dirt_l_level = [280, 282,...]        
        '''
        for i in range(len(self.data.values()[0])):
            print i
            self.x_pose.append(float(self.data['p_x'][i]))
            self.y_pose.append(float(self.data['p_y'][i]))
            self.pose.append([float(self.data['p_x'][i]),
                              float(self.data['p_y'][i])])
            self.dirt_h_level.append(int(self.data['dirt_high_level'][i]))
            self.dirt_l_level.append(int(self.data['dirt_low_level'][i]))
        return self.x_pose, self.y_pose, self.pose, self.dirt_h_level, self.dirt_l_level
    
    def get_laser_ranges(self):
        '''
        Returns a list of laser ranges
        ---
        self.laser_ranges = [[0.0,1.2,1.6,8.6,...],[3.4,1.2,..],...]
        '''
        self.all_laser_ranges = deque()
        for i in range(len(self.data)):
                self.all_laser_ranges.append(self.data[i]['ranges'])        
        self.all_laser_ranges_process = deque()
        for ranges in self.all_laser_ranges:
            ranges_values_list =[]
            for elem in ranges[1:len(ranges)-1].split(','):
                ranges_values_list.append(elem)

            self.all_laser_ranges_process.append(ranges_values_list)
        for elem in self.all_laser_ranges_process:      
            ranges = []
            for i in range(len(elem)):                
                    ranges.append(float(elem[i])) 
            self.laser_ranges.append(ranges)           
        return self.laser_ranges
 
    def get_pose_and_dirt_change(self):
        '''        
        Returns  the pose of the dirt, dirt level change for each sensor, indices in the csv file
        ---
        self.pose_dirt={'h': [ [0.25, 0.5,...],...], 'l':[ [0.25, 0.5,...],...]}         
        self.dirt = {'h': [200,...], 'l':  [200,...]}                       
        self.dirt_indice = {'h': [2,...], 'l':  [3,...]}                             
        self.dirt_num={'h': [1,...], 'l':  [14,...]} 
        '''
        _,_,pose, h_level, l_level = self.get_pose_and_dirt_level()        
        
        self.pose_dirt['h']= deque()
        self.pose_dirt['l']= deque()
        self.dirt['h'] = deque()
        self.dirt['l']=deque()
        self.dirt_indice['h'] = deque()
        self.dirt_indice['l']=deque()
        self.dirt_num['h'] = deque()
        self.dirt_num['l']=deque()
        for indice in range(len(pose)):
            if len(self.dirt.values()[0]) > 0  and len(self.dirt.values()[1])> 0:
                if h_level[indice] != self.dirt['h'][-1]:
                    self.dirt_indice['h'].append(indice)
                    self.pose_dirt['h'].append(pose[indice])
                    self.dirt['h'].append(h_level[indice])
                if l_level[indice]  != self.dirt['l'][-1]:
                    self.dirt_indice['l'].append(indice)
                    self.pose_dirt['l'].append(pose[indice])
                    self.dirt['l'].append(l_level[indice])
            else:
                self.dirt_indice['h'].append(indice)
                self.pose_dirt['h'].append(pose[indice])
                self.dirt['h'].append(h_level[indice])
                self.dirt_indice['l'].append(indice)
                self.pose_dirt['l'].append(pose[indice])
                self.dirt['l'].append(l_level[indice])
        self.dirt_num['h'].append(0)
        self.dirt_num['l'].append(0)            
        for j in range(len(self.dirt['h']) - 1):
                if(self.dirt['h'][j+1] > self.dirt['h'][j]):
                    self.dirt_num['h']. append(self.dirt['h'][j+1] - self.dirt['h'][j] )
                else:
                     self.dirt_num['h']. append(256 - self.dirt['h'][j] + self.dirt['h'][j+1])        
        for j in range(len(self.dirt['l']) - 1):
                if(self.dirt['l'][j+1] > self.dirt['l'][j]):
                    self.dirt_num['l']. append(self.dirt['l'][j+1] - self.dirt['l'][j] )
                else:
                     self.dirt_num['l']. append(256 - self.dirt['l'][j] + self.dirt['l'][j+1])              
        return self.pose_dirt, self.dirt, self.dirt_indice, self.dirt_num
    
    def get_robot_path(self):
         '''
         Visualizing the robot path
         '''
         _, _,pose,_,_= self.get_pose_and_dirt_level()
         pose_duplicate_remove = deque()
         for i in range(len(pose)):   
             if len(pose_duplicate_remove)!=0:
                 x_pose = round(float(pose[i][0]), 4)
                 y_pose = round(float(pose[i][1]), 4)
                 x_dup = round(float(pose_duplicate_remove[-1][0]), 4)
                 y_dup = round(float(pose_duplicate_remove[-1][1]), 4)                 
                 if x_pose - x_dup>0.05  or y_pose - y_dup > 0.05:                  
                     pose_duplicate_remove.append([x_pose, y_pose])                     
             else:
                 x_pose = round(float(pose[i][0]), 4)
                 y_pose = round(float(pose[i][1]), 4)
                 pose_duplicate_remove.append([x_pose, y_pose])                 
         Path().creater(pose_duplicate_remove, color='Green')
    
    def visualize_dirt(self, key='h'):
        '''
        Vizualizing the dirt markers
        '''
        pose,_,_,_ = self.get_pose_and_dirt_change()
        print 'Visualizing  ' + str(key)        
        size = [[dirt_size,dirt_size,0]]* len(pose[key])
        if key=='h':
            color =['Red']* len(pose[key])
        elif key=='l':
            color =['Green']* len(pose[key])        
        VizualMark().publish_marker(pose[key], size, color)
    
    def visualize_grid_dirt(self, key='h'):
        '''
        Vizualizing the dirt grids
        '''
        grid_label_value = self.get_output(key)
        center_list = deque()
        for keys in grid_label_value.keys():
            for i in range(1, len(keys)-1):
                value = keys.split(',')
                center_list.append([float(value[0][1:]), float(value[1][:-1])])        
        pose=deque()
        color=deque()
        for center in center_list:
            if grid_label_value[str(center)] > 0:
                pose.append(center)
                color.append(self.get_color(grid_label_value[str(center)]))
            else:
                continue
        VizualMark().publish_marker(pose=pose, sizes=[[grid_size,grid_size,0.0]]*len(pose), color=color)
      
    def get_color(self,value):        
        if value <=threshold['low']:
            return color['low']
        elif threshold['low']< value <=threshold['medium']:
            return color['medium']
        else:
            return color['high']
    
    def arrange_grid_point(self):
        '''
        Return a dictionary of dirt level in every cell of the map
        ---
        self.grid_point =  {'[0.5,0.5]': [[0.1,0.1],...],...]}
        '''
        center, center_corner_dict = self.get_rectangle_corners()
        _,_,pose, _,_ = self.get_pose_and_dirt_level()
          
        self.pose_used =0

        for key_center in center:
                # print key_center 
                x_list=[]
                y_list =[]  
                self.grid_point[str(key_center)] = deque()
                for value in center_corner_dict[str(key_center)]:                                 
                    x_list.append(value[0])
                    y_list.append(value[1])
                x_min = min(x_list)
                x_max = max(x_list)
                y_min = min(y_list)
                y_max = max(y_list)              
                
                for [x_position, y_position] in pose:
                    if (x_position > x_min and x_position < x_max) and (y_position > y_min and y_position < y_max) :#and  not in self.grid_point[str(key_center)]:
                        if [x_position, y_position] not in self.grid_point[str(key_center)]:
                            self.grid_point[str(key_center)].append([x_position, y_position])
                            # self.pose_used +=1
                        else: continue
                # print self.grid_point[str(key_center)]
                
        return self.grid_point

    def get_output(self, key = 'h'):
        '''
        Return a dictionary of dirt level in every cell of the map
        ---
        self.output =  {'[0.5,0.5]': 10,...}
        '''
        self.output = OrderedDict()
        center_list = self.get_center()
        grid_points_list = self.arrange_grid_point()
        position, dirt, dirt_indice, dirt_num = self.get_pose_and_dirt_change()        
        # indice_used = []      
        
        for center_key, center_grid_point in grid_points_list.items():
            self.output[center_key] = 0
            if center_grid_point:                 
                for coordinates in center_grid_point:
                    for i in range(len(position[key])):
                        if coordinates ==position[key][i]:
                            # indice_used.append(i)
                            self.output[center_key] += dirt_num[key][i]
        # print self.output.values(), sum(self.output.values())
        # print sum(dirt_num[key]), len(dirt_num[key])
        # print len(indice_used)

        return  self.output
    
    def label_output(self, output_list=[]):
        '''
        Return a dictionary of class dirt level in every cell of the map
        ---
        self.labels =  {'[0.5,0.5]': 0.0,...}
        '''
        
        for dirt_val in output_list:
            if dirt_val < dirt_level_threshold:
                self.labels.append(0.0)
            else:
                self.labels.append(1.0)
        return self.labels
        
    def get_input(self):
        '''
        Return a list laser ranges input for training
        ---
        self.input=[[1.0,2.0,4.0,..],...]
        '''
        self.laser_list = self.get_laser_ranges()
        
        inf_val = False
        for i in range(len(self.laser_list)):
            for j in range(len(self.laser_list[i])):
                if self.laser_list[i][j]==float('inf'):
                    inf_val = True
            if not inf_val:
                self.input.append(self.laser_list[i])
            inf_val = False
        return self.input
    
    def extract_train_data(self):
        inputs = self.get_input()
        outputs = self.get_output().values()
        # labels = self.label_output(outputs)
        train_filename =path +'/train'  '/train_dirt_values_' + str(self.file_id) + '.csv'
        fieldnames=['inputs', 'outputs']
        csvcreater(train_filename, fieldnames)
        for i in range(len(inputs)):
            train_input = inputs[i]            
            train_output = outputs
            row_values = [train_input, train_output]
            row_dict = dict(zip(fieldnames,row_values))
            csvwriter(train_filename,fieldnames, row_dict)
            print 'Writting ' + 'training data numero:   '  + str(i)
    
    def view_dirt_cell(self):
        pose_center = self.get_center()
      
        dirt_filename =  path + 'train/' + 'train_' + str(self.file_id)+'.csv'
        print 'Processing {}'.format(dirt_filename)
        process = csvread(dirt_filename)
   
        dirt_list = []
        for elem in process[0]['outputs'][7:-2].split(','):
            dirt_list.append(float(elem))
     
        dirt_pose = []
        for i in range(len(dirt_list)):
            if dirt_list[i] ==1.0:
                dirt_pose.append(pose_center[i])
        VizualMark().publish_marker(dirt_pose, sizes=[[0.15,0.15,0.15]]*len(dirt_pose), color=['Red']*len(dirt_pose))
    
    def view_room_legend(self):
        room_dict = {'Sofa': [0.4,1.9], 'Baxter':[-0.6, -1.8],'TV': [-1.75,2.1], 'Server Room':[2.5,6.0],'Fridge':[1.07,7.6], 'Door':[4.5,2.25]}
        TextMarker().publish_marker(room_dict.keys(), room_dict.values())
                
                        

if __name__ == '__main__':
    rospy.init_node('process')   
    id_num=34
    Process(id_num).view_room_legend()
    Process(id_num).visualize_dirt(key='l')

    # pose = Process().get_center()
    # filename = '/home/mtb/Documents/data/accuracy_prediction.csv'
    # data =  csvread(filename)
    # labels= []
    # predicted =[]
    # for elem in data[-1]['labels'][1:-1].split(','):
    #     labels.append(float(elem))
    # for elem in data[-1]['predicted'][1:-1].split(','):
    #     predicted.append(float(elem))
    # pose_labels, pose_predicted = [],[]
    # for i in range(1296):
    #     if labels[i] ==1.0:
    #         pose_labels.append(pose[i])
    #     if predicted[i]==1.0:
    #         pose_predicted.append(pose[i])
    # print pose_labels, pose_predicted
    # size = [[0.25,0.25,0]] * len(pose_labels)
    # color=['Red'] * len(pose_labels)
    # print pose_labels
    # VizualMark().publish_marker(pose_labels,size, color )
    
    # Process().view_room_legend()
    # Process(id_num= 18).extract_train_data()
    # i = 16
    # fieldnames=['dirt_val']
    # name_path =  '/home/mtb/Documents/data/class/' + 'dirt_file_id_' + str(i)
    # csvcreater( name_path,fieldnames=fieldnames)
    # value_list  = Process(id_num= i).get_output().values()
    # print sum(value_list)
    # for j in range(len(value_list)):
    #     csvwriter(name_path, fieldnames, row= {fieldnames[0]: value_list[j]})

    # Process(id_num= 2).view_dirt_cell()
