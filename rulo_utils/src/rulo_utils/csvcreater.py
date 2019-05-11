#!/usr/bin/env python
import os
import csv

def csvcreater(filepath=None, fieldnames=[]):
    if os.path.isfile(filepath):
        # print ' Already created'
        return True
    else:        
        with open(filepath,  'w') as csvfile:                 
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            return False

