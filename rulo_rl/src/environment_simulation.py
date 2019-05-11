#!/usr/bin/env python
import rospy
from rulo_utils.csvreader import csvreader
from rulo_base.markers import VizualMark
from rulo_utils.csvconverter import csvconverter

filedata = csvreader('/home/mtb/catkin_ws/src/rulo_pkg/rulo_rl/src/markers_2017-08-02.csv')
rospy.init_node('env_viz')
pose = []
color =[]
size = []
value = []
for i in range(len(filedata)):
     pose.append(filedata[i]['pose'])
     size.append(filedata[i]['size'])
     color.append(filedata[i]['color'])
     value.append(filedata[i]['value'])

pose = csvconverter(pose)
color = color
size = csvconverter(size)

print value[0]

VizualMark().publish_marker(pose = pose, color = color , sizes= size)