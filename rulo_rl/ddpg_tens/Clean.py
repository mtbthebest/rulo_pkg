#!/usr/bin/env python
import numpy as np
import rospy
from collections import OrderedDict
from random import randint, choice,sample
import sys
import csv
from rulo_base.markers import VizualMark 
from rulo_utils.csvcreater import csvcreater
from rulo_utils.csvwriter import csvwriter
from rulo_utils.csvreader import csvread
from rulo_utils.csvconverter import csvconverter

indice_dict = {'f':0,'r':1,'b':2,'l':3}
marker_filename = '/home/mtb/rulo_ws/src/rulo_pkg/rulo_rl/ddpg_tens/markers_test_state_2017-08-09.csv'
cleaned_state_filename= '/home/mtb/rulo_ws/src/rulo_pkg/rulo_rl/ddpg_tens/cleaned_state_2017-08-09.csv'
dust_filename = '/home/mtb/rulo_ws/src/rulo_pkg/rulo_rl/ddpg_tens/dust_2017-08-09.csv'
class Clean:
    def __init__(self):       
        self.center_pose =(0,0)
        self.point_num = 10
        self.center_list = list()
        self.total_dust = 0
        self.cleaned_state = []
        self.max_dust_list = []
        csvcreater(cleaned_state_filename, ['state'])

    def reset(self):
        return sample(self.get_pose(), 1)
    
    def get_pose(self):
        self.pose_list = []
        for x in xrange(self.point_num):
            for y in xrange(self.point_num):
                self.center_pose = (x + 0.5, y+0.5, 0)
                self.center_list.append(self.center_pose)
        if not csvcreater(marker_filename, fieldnames=['pose']):
             csvwriter(marker_filename, ['pose'], {'pose':self.center_list})
        return self.center_list
    
    def create_env(self):
        self.get_pose()

        filedata = csvread(marker_filename)
        pose = filedata[0]['pose']
        elem_list = []
        i = 0
        k=1
        while( len(pose) - k >0):
            elem_list.append(pose[k:13+k])
            k +=15
        num_list = []
        pose = []

        for i in range(len(elem_list)):
            k=0
            for num in elem_list[i][1:12].split(','):               
                num_list.append(float(num))
            
            pose.append(num_list)
            num_list =[]

        pose_list  = []
        for pose in pose:
            pose_list.append(pose)

        color =[]
        dust_num = []
        for i in range(len(pose_list)):
            dust_num.append(randint(1,20000))
      
        if not csvcreater(dust_filename, ['dust']):
            csvwriter(dust_filename, ['dust'], {'dust': dust_num})

        for dust_num in dust_num:
            if dust_num > 15000:
                color.append('Red')
            elif dust_num>10000 and dust_num <=15000:
                color.append('Yellow')
            elif dust_num>5000 and dust_num <=10000:
                color.append('Blue')
            else:
                color.append('Gray')

        size = [[1,1,0]] * 100
        # print pose_list

        # VizualMark().publish_marker(pose=pose_list, color=color, sizes=size)
        return pose_list




    def step(self, state, action):        
        done = False
        
        action_val = action[0] 
        if action_val > 0.5:              
            
            # print('Cleaning:  ' + str( self.cleaned_state[-1]))            
            dust_read = list()
            with open(dust_filename, 'r') as csvfile:
                csvread = csv.DictReader(csvfile)
                for row in csvread:
                    dust_read.append(row)
        
            dust_list = []
            for elem in dust_read[0]['dust'][1:-1].split(','):
                dust_list.append(int(elem))           
            
           
            filedata = list()
            with open(marker_filename, 'r') as csvfile:
                csvread = csv.DictReader(csvfile)
                for row in csvread:
                    filedata.append(row)
            pose = filedata[0]['pose']

           
            elem_list = []
            i = 0
            k=1
            while( len(pose) - k >0):
                elem_list.append(pose[k:13+k])
                k +=15
            num_list = []
            pose = []

            for i in range(len(elem_list)):
                k=0
                for num in elem_list[i][1:12].split(','):  
                    num_list.append(float(num))

                pose.append(num_list)
                num_list =[]
            
            state_mod = [state[0], state[1],0.0]
                        
            self.cleaned_state.append(pose.index(state_mod))
            print 'clean: ' + str(self.cleaned_state)
            csvwriter('/home/mtb/rulo_ws/src/rulo_pkg/rulo_rl/ddpg_tens/max_dust_2017-08-09.csv',['max_dust'],  {'max_dust': dust_list[self.cleaned_state[-1]]})
            
            max_dust_reader = list()
            with open('/home/mtb/rulo_ws/src/rulo_pkg/rulo_rl/ddpg_tens/max_dust_2017-08-09.csv', 'r') as csvfile:
                csvread = csv.DictReader(csvfile)
                for row in csvread:
                    max_dust_reader.append(row)          
            
            max_dust =  int(max_dust_reader[-1]['max_dust'])
            if max_dust not in self.max_dust_list:
                self.max_dust_list.append(max_dust)
            
            maximum_dust_val =  max(self.max_dust_list)        
                
         
            if (float(dust_list[self.cleaned_state[-1]]) / float(maximum_dust_val)) >= 0.5:
                reward = 1.0
            if (float(dust_list[self.cleaned_state[-1]]) / float(maximum_dust_val)) < 0.5:
                reward = 0.0
            
            print reward
        
            next_state_with_z = [state[0] + randint(-1,1), state[1] +randint(-1,1),0.0]           
            
            if  next_state_with_z not in pose:
                done = True          
                next_state = state
            else:
                 if next_state_with_z == state_mod:                
                    next_state_with_z = [state[0] + 1, state[1],0.0]
                    if  next_state_with_z not in pose:
                        next_state_with_z = [state[0] -1, state[1],0.0]
                        if  next_state_with_z not in pose:
                           
                            next_state = state
                            done=True
                        else:
                            next_state =np.array(next_state_with_z[0:2])
                    else:
                        next_state =np.array(next_state_with_z[0:2])
                 else:
                    next_state =np.array(next_state_with_z[0:2])
            
            print next_state
               
          
        else:
            reward =0.0
            filedata = list()
            with open(marker_filename, 'r') as csvfile:
                csvread = csv.DictReader(csvfile)
                for row in csvread:
                    filedata.append(row)
            pose = filedata[0]['pose']

           
            elem_list = []
            i = 0
            k=1
            while( len(pose) - k >0):
                elem_list.append(pose[k:13+k])
                k +=15
            num_list = []
            pose = []

            for i in range(len(elem_list)):
                k=0
                for num in elem_list[i][1:12].split(','):  
                    num_list.append(float(num))

                pose.append(num_list)
                num_list =[]

            next_state_with_z = [state[0] + randint(-1,1), state[1] +randint(-1,1),0.0]                   
            if  next_state_with_z not in pose:
                done = True                
                next_state = state
            else:
                 if next_state_with_z == [state[0], state[1], 0.0]:                 
                    next_state_with_z = [state[0] + 1, state[1],0.0]
                    if  next_state_with_z not in pose:                      
                        next_state_with_z = [state[0] -1, state[1],0.0]
                        if  next_state_with_z not in pose:
                            next_state = state
                            done=True
                        else:
                            next_state =np.array(next_state_with_z[0:2])
                    else:
                        next_state =np.array(next_state_with_z[0:2])
                 else:
                    next_state =np.array(next_state_with_z[0:2])
           
        if done:
            
            if self.cleaned_state:
                csvwriter(cleaned_state_filename,['state'], {'state':self.cleaned_state} )  
            self.cleaned_state =[]

                
            
        
        
        return next_state, reward, done
      

        
       
    def done(self):
        sys.exit()
