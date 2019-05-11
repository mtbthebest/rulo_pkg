#!/usr/bin/env	python
import rospy
import os
import numpy as np
from collections import deque, OrderedDict
from rulo_utils.csvreader import csvread
from rulo_utils.csvwriter import csvwriter
from rnn_data import Process
from rulo_base.markers import VizualMark, TextMarker
from visualization_msgs.msg import Marker
import rnn_data
import csv
from nav_msgs.msg import GridCells 
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import Point, Quaternion, Vector3
# from colors import get_color
frame = '/map'
grid_size = 0.25
corner_filename = '/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/data/corners.csv'
class Corners:
    def __init__(self):
        rospy.init_node('corners')
        self.rate = 5
        self.r = rospy.Rate(self.rate) 

    def convert_to_ros_pose(self, pose):
        result = []
        for elem in pose:
            result.append(Point(*(elem[0], elem[1], 0.0)))
        return result
    def extract_corners(self):
        pose =self.convert_to_ros_pose(Process().get_center())
        return pose
    def view_grid(self, cells=[]):
        text_list = [str(list(Process().get_center()).index(elem))
                     for elem in Process().get_center()]
        pose = list(Process().get_center())
        # print pose
        for i in range(700, len(pose)):
            
            VizualMark().publish_marker([pose[i]], sizes=[[0.25,0.25,0.0]],color=['Red'])
            TextMarker().publish_marker(text_list=[text_list[i]], pose =[pose[i]])
            # rospy.sleep(0.25)
        # self.grid_pub = rospy.Publisher('grid_corners', GridCells, queue_size=2)
        # self.grid_marker = GridCells()
        # self.grid_marker.header.frame_id = frame
        # self.grid_marker.header.stamp = rospy.Time.now()

        # self.grid_marker.cell_width = grid_size
        # self.grid_marker.cell_height = grid_size
        # cells = self.extract_corners()

        # self.grid_marker.cells = cells
        # print 'Grid cells'
        # while not rospy.is_shutdown():
        #     self.grid_pub.publish(self.grid_marker)
        #     self.r.sleep()
    
    def check(self):
        pose = list(Process().get_center())
        # print pose       
        # poses = pose[1283:1287]
        # VizualMark().publish_marker(poses, sizes=[[0.25, 0.25, 0.0]] * len(poses), color=['Red']*len(poses))
        # TextMarker().publish_marker(
        #     text_list=[text_list[i]], pose=[pose[i]])
        with open(corner_filename, 'r') as corner_file:
            csvreader = csv.reader(corner_file)
            result_corner = []
            for row in csvreader:
                elem = row[0].split(':')
                result_corner.append([int(elem[0]), int(elem[1])])
                    
        print result_corner

        center_pose = []
        for elem in result_corner:
            for position in pose[elem[0]:elem[1]]:
                center_pose.append(position)
        # print center_pose
        csvwriter('/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/data/pose.csvread',['pose'],[center_pose])

if __name__ == '__main__':
    Corners().check()
