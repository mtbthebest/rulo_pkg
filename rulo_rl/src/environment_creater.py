#!/usr/bin/env python
import numpy as np
import rospy
from collections import OrderedDict
from random import randint, choice,sample
import sys
from rulo_base.markers import VizualMark as markers
from rulo_utils.csvcreater import csvcreater
from rulo_utils.csvwriter import csvwriter

indice_dict = {'f':0,'r':1,'b':2,'l':3}

class EnvCreator:
    def __init__(self):
        rospy.init_node('dust_viz')
        self.center_pose =(0,0)
        self.point_num = 10
        self.total_dust = 0

    def reset(self):
        csvcreater('/home/mtb/rulo_ws/src/rulo_pkg/rulo_rl/src/markers_2017-08-09.csv', ['pose','center','size','color','value','total_dust'])
        self.generate_dust()
       
    def generate_dust(self):
        self.center_list = list()
        self.state_dict =OrderedDict()
        self.target_pose = []
        self.target_size = []
        self.target_color = []
        self.target_value = []
        for x in xrange(self.point_num):
            for y in xrange(self.point_num):
                self.center_pose = (x + 0.5, y+0.5, 0)
                self.center_list.append(self.center_pose)    
        
        for (x,y,z) in self.center_list:          
            self.state_dict[(x,y)] = self.state_creator(x,y)       
        
        for key in self.state_dict.keys():
            self.arrange(self.state_dict[key][0], self.state_dict[key][1])
        
        self.center = list()
        for elem in self.center_list:
            for i in range(4):
                self.center.append(elem)

        for i in range(len(self.target_pose)):
            csvwriter('/home/mtb/rulo_ws/src/rulo_pkg/rulo_rl/src/markers_2017-08-09.csv', ['pose','center','size','color','value','total_dust'],{'pose': self.target_pose[i],'center': self.center[i] ,'size': self.target_size[i] ,'color': self.target_color[i],'value': self.target_value[i],'total_dust': self.total_dust})
        markers().publish_marker(self.target_pose, self.target_size, self.target_color)
        return 'True'

        
    def arrange(self, pose, val):
        for key, value in pose.items():
            if key == 'f' or key == 'b' :
                size = [0.4,0.25,0]
                self.target_pose.append(value)
                self.target_size.append(size)
            if key == 'r' or key == 'l' :
                size = [0.25,0.4,0]
                self.target_pose.append(value)
                self.target_size.append(size)
        
        for value in val:
            self.target_value.append(value)
            self.total_dust +=value
            if value > 18000:
                self.target_color.append('Red')
            elif value > 10000 and value < 18000:
                self.target_color.append('Yellow')
            elif value > 4000 and value < 10000:
                self.target_color.append('Green')
            else:
                self.target_color.append('Gray')

    def state_creator(self, x, y):
            target = OrderedDict()
            dust_val = list()
            target['f'] = (x, y +0.375,0)
            target['r'] = (x+0.375, y,0 )
            target['b'] = (x, y -0.375,0)
            target['l'] = (x-0.375, y, 0)

            for i in range(4):
                dust_val.append(randint(100, 20000))            
            
            return[target,dust_val]
    
if __name__ == '__main__':
    EnvCreator().reset()
    