#!/usr/bin/env python 

import rospy
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import Point, Quaternion, Vector3
from rulo_base.colors import get_color
frame = '/map'

class Path:
    def __init__(self):
        
        # rospy.init_node('robot_path')
        self.rate =1
        self.r = rospy.Rate(self.rate)
        
        self.start = rospy.Time.now()
       
        
    def reset(self):
        self.marker_pub = rospy.Publisher('/line', Marker, queue_size=10)  
        self.marker = Marker()        
        self.marker.action = self.marker.ADD 
        self.marker.header.frame_id = frame     
        self.marker.type = self.marker.LINE_STRIP
        self.marker.lifetime = rospy.Duration(0)    

        self.marker.pose.position = Point(*(0,0,0))
        self.marker.pose.orientation.x = 0
        self.marker.pose.orientation.y = 0
        self.marker.pose.orientation.z =0 
        self.marker.pose.orientation.w = 1             

    def creater(self, pose = [], color='Default'):    
            self.reset()   
            self.marker.id = 0
            self.start = rospy.Time.now()
            while(rospy.Time.now() - self.start < rospy.Duration(5)):                 
                                
                                for position in pose:                     
                                    self.marker.points.append(Point(*(position[0],position[1],0.0)))                                   
                                    self.marker.colors.append(get_color(color))                            
                                    self.marker.id += 1                   
                                    self.marker.header.stamp = rospy.Time.now()     
                                    self.marker.scale = Vector3(*(0.008,0.008,0.0))                       
                                self.marker_pub.publish(self.marker)
                                self.r.sleep()  
        
# if __name__ == '__main__':
    
#         Path().creater([[0,1],[1,2],[0,0]], 'Green')



        
      
   