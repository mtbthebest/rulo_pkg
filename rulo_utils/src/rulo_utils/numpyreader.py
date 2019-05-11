#!/usr/bin/env python
import os
import numpy as np
import csv
from rulo_utils.csvcreater import csvcreater


def numpyreader(filepath):
   
    file = open(filepath, 'rb') 
    data = np.loadtxt(file)
    file.close()
    return data

