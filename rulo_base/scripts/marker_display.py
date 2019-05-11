#!/usr/bin/env python 
import rospy 
from rulo_base.markers import VizualMark

# VizualMark().publish_marker(pose = [[0.875,0.5,0], [2,2,0]], sizes=[[0.25,0.5,0], [1,1,0]],color=['Green', 'Red'])
# VizualMark().publish_marker(pose = [[0.875,0.5,0]], sizes=[[0.25,0.5,0]],color=['Green'], action= 2)

VizualMark().publish_marker(pose = [[2,2,0]], sizes=[[0.5,0.5,0]],color=[ 'Red'])