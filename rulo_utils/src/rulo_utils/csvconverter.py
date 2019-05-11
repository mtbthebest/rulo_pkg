#!/usr/bin/env python
import os
from collections import OrderedDict
data = OrderedDict()


def csvconverter(rows, mode='int'):
    if mode =='int':        
        for row_key in rows.keys():
            data[row_key] = []
            for i in range(len(rows.values()[0])):
                data[row_key].append(int(rows[row_key][i]))        
        return data

def convert_list_to_int(row=[]):
            data = []
            for i in range(len(row[1:-1].split(','))):
                data.append(int(row[1:-1].split(',')[i]))
            return data

def convert_lists_to_int(dict_ ={}):
    data = OrderedDict()
    for i in range(len(dict_)):
        data[str(i)] = []
        data[str(i)].append(convert_list_to_int(dict_[i]))
    return data  
            
            
    
   