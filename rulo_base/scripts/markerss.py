#!/usr/bin/env python 
import numpy as np
import rospy
from visualization_msgs.msg import Marker
from std_msgs.msg import Header
from geometry_msgs.msg import Point, Quaternion

frame = '/base_link'

class VizualMark:
    def __init__(self):
        rospy.init_node('viz')
        self.marker_pub = rospy.Publisher('~markers', Marker, queue_size=5)
        self.rate = 20
        self.r = rospy.Rate(self.rate)
        self.marker = Marker()
        self.marker.header.frame_id = frame
        self.marker.header.stamp = rospy.Time.now()
        self.marker.type = self.marker.CUBE_LIST
        
        self.marker.lifetime = rospy.Duration(0)
        self.marker.id = 0
        self.marker.scale.x =1
        self.marker.scale.y = 1
        self.marker.scale.z = 1
        self.marker.pose.position = Point(1,1,0)
        self.marker.pose.orientation= Quaternion()
        self.marker.color.r= 1
        self.marker.color.g = 0.5
        self.marker.color.b = 0.5
        self.marker.color.a = 1
       
        while not rospy.is_shutdown():
            self.marker.action = self.marker.ADD  
            self.marker.id += 1    
            start = rospy.Time.now()
            position = np.random.choice(5,3)
            point = Point()
            point.x = position[0]
            point.y = position[1]
            point.z = position[2]
            while (rospy.Time.now() - start)< rospy.Duration(2):
                print point
                self.marker.pose.position = point                                
                self.marker_pub.publish(self.marker)
                self.r.sleep()
                

if __name__ == "__main__":
    VizualMark()
        