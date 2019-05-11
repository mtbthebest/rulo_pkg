#!/usr/bin/env python
import rospy
from csvreader import csvread
from rulo_base.markers import VizualMark
from csvconverter import csvconverter

def marker_rviz(filename = None):   
   
    filedata = csvreader(filename)

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
    return VizualMark().publish_marker(pose = pose, color = color , sizes= size)
