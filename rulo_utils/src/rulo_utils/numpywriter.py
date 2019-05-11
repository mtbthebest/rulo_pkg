#!/usr/bin/env python
import os
import numpy as np
import csv
from rulo_utils.csvcreater import csvcreater


def numpywriter(filepath , array=0):
    file = open(filepath, 'ab') 
    np.savetxt(file,array )
    file.close()

