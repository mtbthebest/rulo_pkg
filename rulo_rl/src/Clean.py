#!/usr/bin/env python
import numpy as np
import rospy
from collections import OrderedDict
from random import randint, choice,sample
import sys
from rulo_base.markers import VizualMark as markers
from rulo_utils.csvcreater import csvcreater
from rulo_utils.csvwriter import csvwriter
from rulo_utils.csvreader import csvreader
from rulo_utils.csvconverter import csvconverter

indice_dict = {'f':0,'r':1,'b':2,'l':3}

class Clean:
    def __init__(self):       
        self.center_pose =(0,0)
        self.point_num = 10
        self.center_list = list()
        self.total_dust = 0

    def reset(self):
        for x in xrange(self.point_num):
            for y in xrange(self.point_num):
                self.center_pose = (x + 0.5, y+0.5, 0)
                self.center_list.append(self.center_pose)   
        return sample(self.center_list, 1)

    def step(self, state, action):
        
        done = False
        action_maxval = max(action)         
        action = list(action)
        action_maxval_indice = action.index(action_maxval)        
        # dust_sucked = self.state_dict[tuple(list(state))][1][action_maxval_indice]
        # self.reward.append(dust_sucked)
        filedata= csvreader('/home/mtb/catkin_ws/src/rulo_pkg/rulo_rl/src/markers_2017-08-03.csv')
        pose = []
        center =[]
        size = []
        value = []
        for i in range(len(filedata)):
            pose.append(filedata[i]['pose'])
            size.append(filedata[i]['size'])
            center.append(filedata[i]['center'])
            value.append(filedata[i]['value'])

        pose = csvconverter(pose)
        size = csvconverter(size)
        center = csvconverter(center)
   
        
      
        if action_maxval_indice == 0:
            next_state = [state[0] , state[1]+1]
            dust_pose = [state[0], state[1]+0.375,0] 
            reward =float(value[pose.index(dust_pose)])/20000.0
          

        elif action_maxval_indice == 1:
            next_state = [state[0] + 1 , state[1]]
            dust_pose = [state[0]+0.375, state[1], 0]
            reward =float(value[pose.index(dust_pose)])/20000.0
           
        elif action_maxval_indice == 2:
            next_state = [state[0]  , state[1] -1 ]
            dust_pose = [state[0], state[1]-0.375,0]
            reward = float(value[pose.index(dust_pose)])/20000.0
          
        else:
            next_state = [state[0] -1 , state[1]]
            dust_pose = [state[0] - 0.375, state[1],0]
            reward =float(value[pose.index(dust_pose)])/20000.0
        
        csvwriter('/home/mtb/catkin_ws/src/rulo_pkg/rulo_rl/src/state_2017-08-03.csv',['pose'], {'pose': dust_pose})
        for elem in center:
            if not [next_state[0], next_state[1], 0.0] in center:
                done = True
                break
    
        return next_state, float(reward), done
        
        
        
   
    def done(self):
        sys.exit()

