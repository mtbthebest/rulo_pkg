#!/usr/bin/env python
import os
import csv
import rospy
from geometry_msgs.msg import Point, Pose, Quaternion
from visualization_msgs.msg import Marker
import random
frame = '/map'
class VizualMark:
    def __init__(self):
        rospy.init_node('viz')
        self.marker_pub = rospy.Publisher('~markers', Marker, queue_size=10)
        self.rate = 20
        self.r = rospy.Rate(self.rate)
        self.marker = Marker()
        self.marker.header.frame_id = frame
        self.marker.header.stamp = rospy.Time.now()
        self.marker.type = self.marker.CUBE
        
        self.marker.lifetime = rospy.Duration(0)
        self.marker.id = 0
        # self.marker.scale.x =0.15
        # self.marker.scale.y = 0.15
        # self.marker.scale.z = 0.01
        
        self.marker.pose.orientation= Quaternion()
        self.time_low_level	=[]
        self.num_low_level = []
        self.marker_min_x = []
        self.marker_max_x = []
        self.marker_min_y = []
        self.marker_max_y = []
       
           
        
        
        with open('/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/scripts/values.csv',  'r') as csvfile:
          
            writer = csv.DictReader(csvfile)
            for row in writer:
                self.time_low_level.append(row['time_low_level'])
                self.num_low_level.append(row['num dirt low level'])
                self.marker_min_x.append(row['marker pose min x'])
                self.marker_max_x.append(row['marker pose max x'])
                self.marker_min_y.append(row['marker pose min y'])
                self.marker_max_y.append(row['marker pose max y'])
        while not rospy.is_shutdown():               
                for i in range(len(self.marker_max_x)): 
                            if(self.num_low_level[i] < 50)   :  
                                self.marker.scale.x =0.1
                                self.marker.scale.y = 0.1
                                self.marker.scale.z = 0.01
                                self.marker.color.r= 1
                                self.marker.color.g = 0
                                self.marker.color.b = 0
                                self.marker.color.a = 1   
                            elif(self.num_low_level[i] >= 50 and self.num_low_level[i] < 100)   :  
                                self.marker.scale.x =0.1
                                self.marker.scale.y = 0.1
                                self.marker.scale.z = 0.01
                                self.marker.color.r= 0
                                self.marker.color.g = 1
                                self.marker.color.b = 0
                                self.marker.color.a = 1
                            elif(self.num_low_level[i] >= 100 and self.num_low_level[i] < 200)   :  
                                self.marker.scale.x =0.1
                                self.marker.scale.y = 0.1
                                self.marker.scale.z = 0.01
                                self.marker.color.r= 0
                                self.marker.color.g = 0.75
                                self.marker.color.b = 0.25
                                self.marker.color.a = 1
                            elif(self.num_low_level[i] >= 200 )   :  
                                self.marker.scale.x =0.001
                                self.marker.scale.y = 0.001
                                self.marker.scale.z = 0.01
                                self.marker.color.r= 0
                                self.marker.color.g = 0
                                self.marker.color.b = 1
                                self.marker.color.a = 1
                            else:
                                self.marker.scale.x =0.2
                                self.marker.scale.y = 0.2
                                self.marker.scale.z = 0.01
                                self.marker.color.r= 1
                                self.marker.color.g = 0
                                self.marker.color.b = 0
                                self.marker.color.a = 1
                            self.marker.scale.x =0.1
                            self.marker.scale.y = 0.1
                            self.marker.scale.z = 0.01
                            self.marker.header.frame_id = frame
                            self.marker.header.stamp = rospy.Time.now()
                             
                            self.marker.action = self.marker.ADD  
                            self.marker.id += 1        
                                                          
                            self.marker.pose.position.x = random.choice([float(self.marker_min_x[i]),float(self.marker_max_x[i])])
                            self.marker.pose.position.y= random.choice([float(self.marker_min_y[i]),float(self.marker_max_y[i])])
                            self.marker.pose.position.z  = 0   
                            self.start = rospy.Time.now()
                           
                            self.marker_pub.publish(self.marker)
                            self.r.sleep()
                    

if __name__ == "__main__":
    VizualMark()
        