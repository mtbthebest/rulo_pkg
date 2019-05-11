#!/usr/bin/env python
import csv
import matplotlib.pyplot as plt
step = []
reward = []
with open('./ddpg.csv',  'r') as csvfile:
          
            writer = csv.DictReader(csvfile)
            for row in writer:
               step.append(row['step'])
               reward.append(row['reward'])

plt.plot(step,reward)
plt.show()
