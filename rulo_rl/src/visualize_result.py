#!/usr/bin/env python
import numpy as np
import csv
from rulo_utils.csvreader import csvreader

reward = csvreader('/home/mtb/catkin_ws/src/rulo_pkg/rulo_rl/src/reward_2017-08-02.csv')
print reward
