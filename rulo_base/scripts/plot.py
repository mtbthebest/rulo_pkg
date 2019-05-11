#!/usr/bin/env python
import csv
import matplotlib.pyplot as plt
timer = []
dirt = []
with open('/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/scripts/values.csv',  'r') as csvfile:
          
            writer = csv.DictReader(csvfile)
            for row in writer:
                timer.append(row['time_low_level'])
                dirt.append(row['num dirt low level'])

plt.plot(timer, dirt)
plt.show()
