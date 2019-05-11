#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy

MAX_VEL = 0.2
MIDLE_VEL = 0.2
ROT_VEL = 0.5
REDU_VEL = 0.001

class Joy_controll(object):
    def __init__(self):
        self.rate = rospy.Rate(20)
        # topics
        self.sub = rospy.Subscriber("/rulo/joy", Joy, self.callback, queue_size=10)
        self.base = rospy.Publisher("/Rulo/cmd_vel",Twist,queue_size=10)

    def callback(self,data):
        vel_order = Twist()
        if data.buttons[10] == 1 or data.buttons[8] == 1:
            vel_max = MIDLE_VEL
            rot = ROT_VEL
            if data.buttons[8] == 1:
                vel_max = MAX_VEL
            vel_x = data.axes[3] * vel_max
            vel_y = data.axes[2] * vel_max
            vel_w = data.axes[0] * rot

            if np.sqrt(vel_x**2 + vel_y**2) > vel_max:
                vel = np.sqrt(vel_x**2 + vel_y**2)
                vel_x = vel_x / vel * vel_max
                vel_y = vel_y / vel * vel_max

            if np.sqrt(vel_x**2 + vel_y**2 + vel_w**2) != 0.0:
                vel_order.linear.x = vel_x
                vel_order.linear.y = vel_y
                vel_order.angular.z = vel_w
                
            self.base.publish(vel_order)
    def run(self):
        while not rospy.is_shutdown():
            self.rate.sleep()

if __name__ == '__main__':
    rospy.init_node('base_controller')
    a = Joy_controll()
    a.run()
