#!/usr/bin/env python

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
marker_filename = '/home/mtb/rulo_ws/src/rulo_pkg/rulo_rl/ddpg_tens/data/markers_test_state_2017-08-09.csv'

dust_filename = '/home/mtb/rulo_ws/src/rulo_pkg/rulo_rl/ddpg_tens/data/dust_2017-08-09.csv'

class Clean:
    def __init__(self):       
        self.center_pose =(0,0)
        self.point_num = 10
        self.center_list = list()
        self.total_dust = 0
        self.cleaned_state = []
        self.max_dust_list = []
      

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

        dust_list = csvread(dust_filename)
    
       
        for elem in dust_list[1]['dust'][1:-1].split(','):
            dust_num.append(int(elem))
        print dust_num

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
        print color

        VizualMark().publish_marker(pose=pose_list, color=color, sizes=size)
        clean_space = [[ 2.5, 6.5, 0.0], [ 3.5,6.5, 0.0],[ 4.5, 6.5, 0.0],[ 5.5 , 5.5,  0.0],[ 6.5, 5.5, 0.0],
        [ 7.5, 5.5, 0.0],
        [ 6.5, 4.5,  0.0],[ 6.5, 3.5, 0.0],[ 5.5,  4.5, 0.0],[ 7.5 , 4.5, 0.0],[ 8.5, 5.5, 0.0],[ 7.5, 5.5, 0.0],
        [ 7.5, 4.5, 0.0],[ 6.5,  3.5, 0.0],[ 4.5, 3.5, 0.0],[ 5.5 , 1.5, 0.0],[ 7.5  ,2.5, 0.0],[ 6.5 , 3.5, 0.0],
        [ 3.5, 2.5, 0.0],[ 4.5 ,2.5,  0.0],[ 5.5, 2.5, 0.0],[ 6.5, 2.5, 0.0],[ 7.5 ,1.5, 0.0],[ 6.5 ,1.5, 0.0],
        [ 6.5 ,4.5,  0.0], [4.5 ,6.5,0.0], [ 3.5,5.5, 0.0],[ 3.5 ,4.5, 0.0],[ 4.5 ,5.5 , 0.0],[ 0.5, 0.5, 0.0],[ 1.5 ,1.5, 0.0],
        [ 2.5,3.5, 0.0],[ 4.5, 4.5, 0.0]]
        # return pose_list
        pose_list_updated = []
        indice_list = []
        for elem in pose_list:
            if elem not  in clean_space:
                pose_list_updated.append(elem)
                indice_list.append(pose_list.index(elem))
        
       
        new_color_list = []
        for elem in indice_list:
            new_color_list.append(color[elem])
        print len(new_color_list)
        size = [[1,1,0]] * 73
        VizualMark().publish_marker(pose=pose_list_updated, color=new_color_list, sizes=size)
if __name__ == '__main__':
    Clean().create_env()