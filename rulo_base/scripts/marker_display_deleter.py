#!/usr/bin/env python 
import rospy 
from rulo_base.markers import VizualMark

rospy.init_node('display')
VizualMark().publish_marker(pose = [[0.875,0.5,0]], sizes=[[0.25,0.5,0]],color=['Green'], action= 2)