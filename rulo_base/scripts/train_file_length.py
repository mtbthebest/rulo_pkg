#!/usr/bin/env python
import os

import csv
from collections import OrderedDict, deque
import random
from math import pow
import numpy as np
from rulo_utils.csvreader import csvread

path =  os.path.abspath(os.path.dirname('/home/mtb/Documents/data/train/'))
from rulo_utils.csvcreater import csvcreater
from rulo_utils.csvwriter import csvwriter

all_files = []
data_list = []
length_files = []
for files in os.listdir(path):
    if files.startswith('train_dirt_values_'):
        all_files.append(files)
        data = csvread(path +'/'+ files)
        data_list.append(len(data))

        try:


            length_files.append(data_list[-1] - data_list[-2] +1)
            
        except:
            length_files.append(data_list[-1])
    print length_files[-1]

csvcreater(path+'/' + 'length_dirt_val.csv', fieldnames=all_files)
csvwriter(path+'/' + 'length_dirt_val.csv', all_files, dict(zip(all_files, length_files)))
