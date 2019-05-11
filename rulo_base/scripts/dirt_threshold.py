#!/usr/bin/env	python
import	rospy
import numpy as np
from collections import deque, OrderedDict
from rulo_utils.csvreader import csvread
from rulo_utils.csvcreater import csvcreater
from rulo_utils.csvwriter import csvwriter




path = '/home/mtb/Documents/data/class/'
name = 'dirt_file_id_16'
data = csvread(path + name)
value = []
val_len = []
for i in range(len(data)):
   val = int(data[i]['dirt_val'])
   val_len.append(val)
   if val!=0:
       value.append(val)

print sorted(value, reverse=True), len(val_len), sum(value)