#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import numpy as np
from sensor_msgs.msg import Joy
from std_msgs.msg import String
from rulo_msgs.msg import BrushesPWM_cmd
import os


class Joy_controll(object):
    def __init__(self):
        self.rate = rospy.Rate(10)
        # topics
        self.sub = rospy.Subscriber("/rulo/joy", Joy, self.callback, queue_size=10)
        self.sweep = rospy.Publisher("/mobile_base/command/brushesPWM_cmd",  BrushesPWM_cmd, queue_size=10)
        self.alert = rospy.Publisher("/mobile_base/command/mode",  String, queue_size=10)
        self.sweep_flag = False

        
    def callback(self,data):
        if data.buttons[1] == 1:
            if not self.sweep_flag:
                b =  BrushesPWM_cmd()
                b.main_brush = 50
                b.side_brush = 50
                b.vacuum = 50
                self.sweep.publish(b)
                self.sweep_flag = True

## 改良            
        elif data.buttons[2] == 1:
            if self.sweep_flag:
                b =  BrushesPWM_cmd()
                b.main_brush = 0
                b.side_brush = 0
                b.vacuum = 0
                self.sweep.publish(b)
                self.sweep_flag = False

        elif data.buttons[3] == 1:
            s = String()
            s.data = "normal"
            self.alert.publish(s)
            rospy.sleep(1.0)


        
 
    def run(self):
        while not rospy.is_shutdown():
            self.show_interface()
            self.rate.sleep()
        
    def show_interface(self, ):
        os.system("clear")
        print "Please edit"

                

if __name__ == '__main__':
    rospy.init_node('joy_interface')
    a = Joy_controll()
    a.run()
