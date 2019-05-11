#!/usr/bin/env python
import os
import rospy
from rulo_msgs.msg import   AirCondition
from std_msgs.msg import Header
import time
from geometry_msgs.msg import Point, Quaternion, Pose
import tf
from tf.listener import TransformListener
import csv 
from collections import OrderedDict
target_frame = '/map'
source_frame = '/base_footprint'

class PM:
    def __init__(self):
        self.time, self.min= time.localtime()[3], time.localtime()[4]
        rospy.init_node('dirt')
        self.mass =[]
        self.large = []
        self.small = []
        self.dust=[]
        self.tf = OrderedDict()
        self.tf["x"] = []
        self.tf["y"] = []

        rospy.on_shutdown(self.shutdown)
        self.tf_listener = TransformListener()
        rospy.sleep(2.0)
        try: 
            self.tf_listener.waitForTransform(target_frame, source_frame, rospy.Time(), rospy.Duration(1.0))
        except (tf.Exception, tf.ConnectivityException, tf.LookupException):
            rospy.signal_shutdown('tf exception')
      
        self.start = rospy.Time.now()
        rospy.Subscriber('/mobile_base/event/air_condition', AirCondition, self.pm_call)
        rospy.loginfo("start")
        rospy.spin()
        rospy.loginfo("end")       
        
    def pm_call(self, msg):
            self.mass.append(msg.mem1)
            self.large.append(msg.mem2)
            self.small.append(msg.mem3)
            self.dust.append(msg.mem4)
            trans, rot = self.tf_listener.lookupTransform(target_frame, source_frame, rospy.Time())
            self.tf["x"].append(trans[0])
            self.tf["y"].append(trans[1])
            print self.mass, self.large, self.small, self.dust
            print self.tf
    def shutdown(self):
        with open('/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/scripts/pm25.csv',  'w') as self.csvfile:
            self.fieldnames = ['Mass', 'Large','Small','Dust','Position']
            self.writer = csv.DictWriter(self.csvfile, fieldnames=self.fieldnames)
            self.writer.writeheader()
      
            self.writer.writerow({'Mass': self.mass, 'Large': self.large,'Small': self.small, 'Dust': self.dust, 'Position':self.tf })
           
    

if __name__=="__main__":
    PM()