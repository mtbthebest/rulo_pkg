#!/usr/bin/env python 

import rospy
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import Point, Quaternion, Vector3
from tf.transformations import quaternion_from_euler
from colors import get_color
frame = '/map'

class VizualMark:
    def __init__(self):
        
        # rospy.init_node('markers')
        self.rate =50
        self.r = rospy.Rate(self.rate)
        self.marker_dict = dict()
        self.marker_list = list()    
        self.features_dict = dict()
        self.start = rospy.Time.now()
        self.indice = 0
        self.id = 0
    def reset(self,action='add'):
        self.marker_pub = rospy.Publisher('~marker_cube', Marker, queue_size=10)  
        self.marker = Marker()      
        if action== 'add':     
            self.marker.action = self.marker.ADD 
        else:
            self.marker.action = self.marker.DELETE
        self.marker.header.frame_id = frame     
        self.marker.type = self.marker.CUBE_LIST
        self.marker.lifetime = rospy.Duration(0)    

        self.marker.pose.position = Point(*(0,0,0))
        self.marker.pose.orientation.x = 0
        self.marker.pose.orientation.y = 0
        self.marker.pose.orientation.z =0 
        self.marker.pose.orientation.w = 1             

    def publish_marker(self, pose = [], sizes=[[0.25,0.25,0.25]],color=[],action='add' ,convert= False, texture=None,publish_num = 2,id_list = []):    
            print 'Visualization start' 
            self.reset(action)   
            self.marker.id = self.id
            self.start = rospy.Time.now()
            while not rospy.is_shutdown():
                            for j in range(publish_num):
                # while(rospy.Time.now() - self.start < rospy.Duration(30)):                                                              
                                self.scale = []
                                self.color = []                   
                                for i in range(len(pose)):  
                                    # print i
                                    # print pose[i]                   
                                    self.marker.points = [Point(*(pose[i][0], pose[i][1],0.0))]
                                    self.marker.scale = Vector3(*(sizes[i][0], sizes[i][1],sizes[i][2]))
                                    self.marker.color = get_color(color= color[i], convert= convert, texture=texture)                             
                                    self.marker.id = id_list[i]                 
                                    self.marker.header.stamp = rospy.Time.now()                            
                                    self.marker_pub.publish(self.marker)
                                    self.r.sleep()  
                                #     if i ==len(pose) -1:
                                #         print 'terminate'
                                #         self.id = self.marker.id
                            break
                # break
    
    # def delete_marker(self, pose = [], sizes=[[0.25,0.25,0.25]],color=[], convert= False, texture=None):
        
                                        
class TextMarker:
    def __init__(self):
        
        # rospy.init_node('markers')
        self.rate =50
        self.r = rospy.Rate(self.rate)
        self.marker_dict = dict()
        self.marker_list = list()    
        self.features_dict = dict()
        self.start = rospy.Time.now()
        self.indice = 0
        
    def reset(self):
        self.marker_pub = rospy.Publisher('~marker_text', Marker, queue_size=10)  
        self.marker = Marker()        
        self.marker.action = self.marker.ADD 
        self.marker.header.frame_id = frame     
        self.marker.type = self.marker.TEXT_VIEW_FACING
        self.marker.lifetime = rospy.Duration(0)    

        # self.marker.pose.position = Point(*(0.4,1.9,0))
        # self.marker.pose.orientation.x = 0
        # self.marker.pose.orientation.y = 0
        # self.marker.pose.orientation.z =0 
        # self.marker.pose.orientation.w = 1             
        
        self.marker.color = get_color(color= 'Black')   
    def publish_marker(self, text_list=[], pose=[],angle=[]):    
            self.reset()   
            self.marker.id = 0
            self.start = rospy.Time.now()
       
            while not rospy.is_shutdown():
                while(rospy.Time.now() - self.start < rospy.Duration(0.5)):                                                              
                                self.scale = []
                                self.color = []                   
                                for i in range(len(pose)):                     
                                    self.marker.pose.position = Point(*(pose[i][0], pose[i][1],0.0))     
                                    self.marker.pose.orientation = Quaternion(*quaternion_from_euler(1.57,1.57,1.57,axes='sxyz'))
                                    self.marker.scale = Vector3(*(1.0, 1.25,0.3))
                                    self.marker.text = text_list[i]                                                                            
                                    self.marker.id = i
                                    self.marker.header.stamp = rospy.Time.now()                            
                                    self.marker_pub.publish(self.marker)
                                    self.r.sleep()  
                                    if i ==len(pose) -1:
                                        print 'terminate'
                                    
                else:
                        break

class Line:
    def __init__(self):
        
        # rospy.init_node('markers')
        self.rate =50
        self.r = rospy.Rate(self.rate)
        self.marker_dict = dict()
        self.marker_list = list()    
        self.features_dict = dict()
        self.start = rospy.Time.now()
        self.indice = 0
        
    def reset(self):
        self.marker_pub = rospy.Publisher('~marker_line', Marker, queue_size=10)  
        self.marker = Marker()        
        self.marker.action = self.marker.ADD 
        self.marker.header.frame_id = frame     
        self.marker.type = self.marker.LINE_STRIP
        self.marker.lifetime = rospy.Duration(0)    

        # self.marker.pose.position = Point(*(0.4,1.9,0))
        # self.marker.pose.orientation.x = 0
        # self.marker.pose.orientation.y = 0
        # self.marker.pose.orientation.z =0 
        # self.marker.pose.orientation.w = 1             
        self.marker.scale = Vector3(*(0.05, 1.75,5.0))
        
    def publish_marker(self, pose=[],id_list=[],color='Red',duration=60.0):    
            self.reset()   
            self.marker.color = get_color(color= color)   
            # self.marker.id = 0
            self.start = rospy.Time.now()
            while(rospy.Time.now() - self.start < rospy.Duration(duration)): 
                # while not rospy.is_shutdown():
                                                                             
                    self.scale = []
                    self.color = []                   
                    for i in range(len(pose)):                     
                        self.marker.points.append(Point(*(pose[i][0], pose[i][1],0.0)))    
                        # self.marker.text = text_list[i]                                                                            
                        self.marker.id = id_list[i]
                        self.marker.header.stamp = rospy.Time.now()                            
                        self.marker_pub.publish(self.marker)
                        self.r.sleep()  
                        # if i ==len(pose) -1:
                        #     print 'terminate'
                                    
                # else:
                #         break

# if __name__ == '__main__':
#     rospy.init_node('path')
#     Line().publish_marker([[0.0,0.0],[1.0,2.0],[4.0,5.0]])
    