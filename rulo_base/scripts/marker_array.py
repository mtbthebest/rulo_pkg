#!/usr/bin/env python 
import numpy as np
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header
from geometry_msgs.msg import Point, Quaternion

frame = '/map'

class VizualMark:
    def __init__(self):
        rospy.init_node('viz')
        self.marker_pub = rospy.Publisher('~marker_array', Marker, queue_size=10)
        self.rate = 2
        self.r = rospy.Rate(self.rate)
        self.marker = Marker()
        self.marker.action = self.marker.ADD 
        self.marker.header.frame_id = frame     
        self.marker.type = self.marker.CUBE_LIST
        self.marker.lifetime = rospy.Duration(0)
        self.marker.id = 0
        self.marker.scale.x = 0.5
        self.marker.scale.y = 0.5
        self.marker.scale.z = 0
        self.marker.text = '240'
   
        self.marker.pose.orientation.x = 0
        self.marker.pose.orientation.y = 0
        self.marker.pose.orientation.z = 0
        self.marker.pose.orientation.w = 1
        self.marker.pose.position.x =0
        self.marker.pose.position.y = 0
        self.marker.pose.position.z = 0
        self.marker.color.r= 0.5
        self.marker.color.g = 1
        self.marker.color.b = 0.1
        self.marker.color.a = 1

        self.marker.points.append(Point(1, 1,0))
        while not rospy.is_shutdown():
            self.marker.header.stamp = rospy.Time.now()            
           
            self.marker_pub.publish(self.marker)
            self.r.sleep()                        
                                      
            
            

if __name__ == "__main__":
    VizualMark()
        