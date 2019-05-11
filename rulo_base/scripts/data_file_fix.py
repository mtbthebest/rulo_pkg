#!/usr/bin/env	python
import	rospy
import numpy as np
from collections import deque, OrderedDict
from rulo_utils.csvreader import csvread
from rulo_utils.csvcreater import csvcreater
from rulo_utils.csvwriter import csvwriter

path1 = '/home/mtb/Documents/data/test2017-10-17_18h30_2.csv'
path1_fix = '/home/mtb/Documents/data/test2017-10-17_18h30_fix.csv'

data = csvread(path1)
fieldnames =['p_x','p_y','p_z','q_x','q_y', 'q_z', 'q_w','dirt_high_level', 'dirt_low_level', 'wall_time', 'duration',
             'angle_min','angle_max','angle_increment','time_increment','scan_time','range_min','range_max','ranges','intensities','wall_time','duration']


for i in range(len(data)):
    row_val = deque()
    for elem in fieldnames:
        row_val.append(data[i][elem])
    row_dict = dict(zip(fieldnames,row_val))
    csvwriter(path1_fix, fieldnames, row = row_dict)