#!/usr/bin/env python
import os
import rospy
import csv
from collections import OrderedDict, deque
import random
from math import pow
import numpy as np
from rulo_utils.csvreader import csvread
# rospy.init_node('time')   
# start = rospy.get_time()
# starts = rospy.Time.now()
# # print start
# while rospy.Time.now() - starts < rospy.Duration(5):
#     continue
# print (rospy.get_time() - start)

# with open('/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/scripts/values.csv', 'wb') as f:
#     valuecsv = csv.writer(f, delimiter=' ' )
#     valuecsv.writerow(['name', 'frst'])
#     valuecsv.writerow([5,7])
# import csv

# with open('/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/scripts/values.csv',  'w') as csvfile:
#     fieldnames = ['first_name', 'last_name']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

#     writer.writeheader()
#     writer.writerow({'first_name': 'Baked', 'last_name': 'Beans'})
#     writer.writerow({'first_name': 'Lovely', 'last_name': 'Spam'})
#     writer.writerow({'first_name': 'Wonderful', 'last_name': 'Spam'})


# test = [5,4,6]
# par = [43,4]
# if 5 in {set(test), set(par)}:
#      print 'true'
# a =OrderedDict()
# x = {'a':[1 ,2,3,5],'b':[4,5,6,5]}
# for key in sorted(a):
#     print key
#     a[key] = a[key]

# # print a
# a = [1,2,3]
# print max(x.values()[0])

# import time
# print time.localtime()[3]
#print (7%2)

# with open('/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/scripts/values.csv',  'r') as csvfile:
    
#     writer = csv.DictReader(csvfile)
#     for row in writer:
#         print row['marker pose min x']


# print random.uniform(1.55,1.66) - 5

# def cal(*args):
#     print args[0]

# hu= (1,2,3)
# cal(hu)

# import tensorflow as tf
# X = tf.placeholder(dtype = tf.float32, shape=None)

# def angle(angle):
#     try:
#         a =pow((0.708218042904 /0.9999999998175838), 90/angle)  * (0.9999999998175838)
#         print a
#     except ZeroDivisionError:
#         a = 0
#         print a

# angle(0)
# a = np.linspace(-1.65, 9.0, num=int(10.65/0.3))
# a = np.linspace(-4, 2, num=int(6/0.3))
# print a[2]

# def csv_writer(dic, i):
#     with open('/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/scripts/test.csv',  'a') as csvfile:
#         fieldnames = ['first_name', 'last_name']
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         if (i ==0):
#             writer.writeheader()
#         i +=1            
#         writer.writerow(dic)


# csv_writer({'first_name': 'Lovely', 'last_name': 'Spam'}, i=1)

# def call(**kwargs):
#     for k,v in kwargs.values():
#         print v

# a={'a':[(2,4), 'q'], 'b':[(1,1), 't']}
# call(**a)

# print np.asarray(random.sample(xrange(0,20000), 4))
# print np.random.uniform(0,1, size=(1,4))
# a =np.random.randint(-1,1,size=(2,))
# a = np.array([2,4]) +  np.random.randint(-1,1,size=(2,))
# a = [5,4]
# if 4 not in a:
#     print True

# a = np.array([1,2])
# print np.reshape(a,(1,2))
# a = ['a', 'b']
# b =[5,4]
# print dict(zip(a,b))
# b = []
# a =[5,4,3]
# b.append(a)
# b= []
# c = [4,5,6]
# b.append(c)
# print b

# a =deque()
# a.append(0)
# a.append(1)
# a.append(10)
# a.append(10)
# for i in range(len(a)):
#     if a[i]==10:
#         print i

# a =[[1,2],[4,5,6],[6,7,8]]
# b = np.asarray(a)
# print b
# data = csvread('/home/mtb/Documents/data/train_2.csv')
# element = []


# value = data[0]['inputs'][1:-1].split(' ')
# values =[]

# for val in value:
#     if not val == '':
#         if '\n' in val:
#             val.replace('\n', '')
        
#         values.append(float(val))

# for val in values:
#     if val
# print value
# print (values)

# print     len(data[0]['inputs'][2:-1].split('  '))# len(data[0]['inputs'])

# for elem in data[0]['inputs']:
#     if elem not in element:
#         element.append(elem)
# print element

# output = data[0]['outputs'][1:-1].split(',')
# print len(output)

# outputs = []

# for elem in output:
#     outputs.append(int(elem))
# print outputs.index(892)

# filename = '/home/mtb/rulo_ws/src/rulo_pkg/rulo_ml/model/train/evaluation.csv'
# data = csvread(filename)
# dirt = list()
# for i in range(len(data['cells'])):
#     dirt.append(data['cells'][i])
# a = []
# for i in dirt[0][1:-1].split(','):
#     if i != ' dtype=float32)':
#         a.append(i)
# print len(a)
# k =0
# c = OrderedDict()
# for k in range(2):
#     c [str(k)] = []
#     for i in  range(k*5, 5*(k+1)):
#         print a[i]
#         if '  array([ ' in a[i]:
#             elem = (a[i][8:])
#             print elem
        # else:
        #   elem = float(a[i])
        
          
# print c
        
        

# for i in range(len(data.values())):
#     dirt.append(int(data['grid_dirt'][i]))


# a = [1,2,3,5]

# b = [4,3,2]

# print a
# for elem in b:
    
#     if elem in a:
#         print elem
#         a.pop(a.index(elem))
#         print a

# a.append(2)
# print a


a = np.load( '/home/mtb/rulo_ws/src/rulo_pkg/rulo_ml/model/train/evaluation_1.npy')
print a[1]
