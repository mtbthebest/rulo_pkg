#!/usr/bin/env	python
import os
import numpy as np
from collections import deque, OrderedDict
from rulo_utils.csvreader import csvread
from rulo_utils.csvwriter import csvwriter
from rulo_utils.csvconverter import csvconverter, convert_list_to_int, convert_lists_to_int
path = '/home/mtb/Documents/data/'
data_path = '/media/mtb/Data Disk/data/'
filename = ['test2017-12-20_20h30_1.csv',
            'test2017-12-20_20h30_2.csv'
            ]

class Rewrite():
    def __init__(self):
        self.data = OrderedDict()
        self.filename=[]
        for i in range(len(filename)):
            self.filename.append(path+filename[i])
        print self.filename
    def rewrite(self):   
           
        for files in self.filename:
            print self.filename.index(files)
            data = csvread(files)
            cells_list = []
            for keys, values in data.items():
                print keys, len(values)
                cells_list.append(len(values))
            
            max_cells = min(cells_list)
            data_list =[data.values()[i][:max_cells] for i in range(len(data.values()))]
            csvwriter(data_path + filename[0] , data.keys() , data_list)
            print 'Wrote'
    
    def check(self):
        data = csvread(data_path + filename[0])
        for keys in data.keys():
            print len(data[keys])
        # print len(self.data.values())
        # print self.data.keys()
        # print len(self.data.values()[0][0] +self.data.values()[0][1]) 
      
        # for keys self.data.keys
        #     csvwriter(data_path+filename[0], self.data.keys(),self.data.values()[i])

            

if __name__ == '__main__':
    Rewrite().rewrite()
    # Rewrite().check()
