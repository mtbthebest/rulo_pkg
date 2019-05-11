#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Point
from rulo_base.markers import VizualMark

if __name__ == '__main__':
    rospy.init_node('marker_sim')
    VizualMark().publish_marker(pose = [[0.875,0.5,0]], sizes=[[0.25,0.5,0]],color=['Green'])
  