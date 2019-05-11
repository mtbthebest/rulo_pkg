#!/usr/bin/env python
import os
import sys
      
from rulo_base.Rulo import Rulo as rulo
import numpy as np
from geometry_msgs.msg import Twist
import rospy

import random
from math import radians
angle_forward = 0.8
angle_right = 0.16
angle_backward =-0.64
if __name__ == '__main__':
        rospy.init_node("rotate", anonymous=True)
        
        #rulo().nav_tf([0.48,-0.22,radians(90)])
        x_position =2.09
        y_position = -2.65
        while not rospy.is_shutdown():
                rulo().nav_tf([x_position,y_position ,radians(120)])
                rospy.sleep(2.0)
                x_position = np.linspace(2.09, 4.0, num=int(4/0.3))
                y_position  = np.linspace(-2.65, 7, num=int(7/0.3))
                state = 0
                for x_pose in x_position:
                        for y_pose in y_position:
                                rulo().nav_tf([x_pose, y_pose,radians(120)])
                                state +=1
                                rospy.loginfo(str(x_pose) + str( " ," ) + str(y_pose))
                                print state


        # rulo().nav_tf([-0.327,-2.3,0.986])
        # x_position = random.triangular(-4.5, 7.5)
        # y_position = random.triangular(0, -5.5)
        # angle = 0
        # for i in range(10):
        #         print x_position, y_position
        #         rulo().nav_tf((x_porosrun rulo keyboard_nodesition, y_position, angle))
        # rulo().rotate_with_vel()