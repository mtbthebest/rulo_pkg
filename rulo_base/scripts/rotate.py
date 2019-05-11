#!/usr/bin/env python

""" nav_test.py - Version 1.1 2013-12-20

    Command a robot to move autonomously among a number of goal locations defined in the map frame.
    On each round, select a new random sequence of locations, then attempt to move to each location
    in succession.  Keep track of success rate, time elapsed, and total distance traveled.

    Created for the Pi Robot Project: http://www.pirobot.org
    Copyright (c) 2012 Patrick Goebel.  All rights reserved.

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.5
    
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details at:
    
    http://www.gnu.org/licenses/gpl.html
      
"""

import rospy
import actionlib
from actionlib_msgs.msg import *
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped, Point, Quaternion, Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from random import sample
from math import pow, sqrt
import tf

class Rulo():
    def __init__(self):
        rospy.init_node('nav_test', anonymous=True)
        
        rospy.on_shutdown(self.shutdown)
        self.rate = 20
        self.r = rospy.Rate(self.rate)
        # How long in seconds should the robot pause at each location?
        self.rest_time = rospy.get_param("~rest_time", 10)
        
        # Are we running in the fake simulator?
        self.fake_test = rospy.get_param("~fake_test", False)
      
        
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        
        
        self.move_cmd = Twist()    

    def rotate(self,angle):
        self.move_cmd.linear.x = 0
        self.move_cmd.linear.y = 0
        self.move_cmd.linear.z = 0
        self.move_cmd.angular.x = 0
        self.move_cmd.angular.y = 0
        self.move_cmd.angular.z = angle
        start_angle = 0
        while not rospy.is_shutdown():
            self.cmd_vel_pub.publish(self.move_cmd)
            self.r.sleep()

    def dirt_detect(self):
        pass
            
    
            
   

    def shutdown(self):
        rospy.loginfo("Stopping the robot...")
        self.move_base.cancel_goal()
        rospy.sleep(2)
        self.cmd_vel_pub.publish(Twist())
        rospy.sleep(1)
      


# if __name__ == '__main__':
#     try:
#         Rulo().nav_tf(1,1,0)
#     except rospy.ROSInterruptException:
#         rospy.loginfo("Navigation test finished.")