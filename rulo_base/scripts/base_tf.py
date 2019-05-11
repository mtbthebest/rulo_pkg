#!/usr/bin/env python 
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
import tf
from tf.transformations import euler_from_quaternion
from tf.listener import TransformListener
from std_msgs.msg import String, Header
from math import degrees
target_frame = '/map'
source_frame = '/base_footprint'

class TF():
    def __init__(self):
        rospy.init_node('transform')
        self.rate = 50
        self.r = rospy.Rate(self.rate)
        self.tf_listener = TransformListener()
        rospy.sleep(1.0)
        self.tf_pub = rospy.Publisher("tf_pub", PoseStamped, queue_size=10)
        # self.tf_rot = rospy.Publisher("tf_rot", Quaternion, queue_size=10)
        self.point = PoseStamped()
        self.list = list()
        try: 
            self.tf_listener.waitForTransform(target_frame, source_frame, rospy.Time(), rospy.Duration(1.0))
        except (tf.Exception, tf.ConnectivityException, tf.LookupException):
            rospy.signal_shutdown('tf exception')
        
        while not rospy.is_shutdown():
            self.point.header = Header()
            self.point.header.stamp = rospy.Time.now()
            trans, rot = self.tf_listener.lookupTransform(target_frame, source_frame, rospy.Time())
            
            self.point.pose = Pose(Point(*trans), Quaternion(*rot))
        
            self.tf_pub.publish(self.point)          
          
         
            self.r.sleep()


if __name__ == '__main__':
    TF()