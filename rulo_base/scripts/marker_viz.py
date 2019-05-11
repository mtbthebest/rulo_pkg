#!/usr/bin/env python 
import numpy as np
import rospy
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import Point, Quaternion
# from rulo_base.csvreader import readfile
# from rulo_base.colors import get_color
color_name = ['Yellow', 'Red', 'Blue','Red', 'Gray']

marker_dict = dict()
marker_dict['Yellow'] =[]
marker_dict['Blue'] =[]
marker_dict['Red'] =[]
frame = '/map'
marker_dict['Gray'] =[]
def get_pose(pose_x, pose_y):
    return Point(*(pose_x, pose_y , 0 ))

if __name__ == "__main__":
    data=
    rospy.init_node('PATH')
    marker_pub = rospy.Publisher('/marker_cube', Marker, queue_size=10)
    markers = Marker()

    
    markers.action = markers.ADD
    markers.header.frame_id = frame
    markers.type = markers.LINE_STRIP
    markers.lifetime = rospy.Duration(0)
    markers.id = 0
    markers.scale.x = 0.025
    markers.scale.y = 0
    markers.scale.z = 0


    markers.points =[Point(*(0.0, 0.0,0.0)),Point(*(1.0, 1.0,0.0))]
    markers.color.r = 0.5
    markers.color.g = 1
    markers.color.b = 0.1
    markers.color.a = 1
    # for i in range(len(values)):
    #     if int(values[i]['num dirt low level']) < 3000:
    #         marker_dict['Gray'].append(get_pose(float(values[i]['marker pose x']), float(values[i]['marker pose y'])))
    #     if int(values[i]['num dirt low level']) >= 3000 and int(values[i]['num dirt low level']) < 6000 :
    #         marker_dict['Blue'].append(get_pose(float(values[i]['marker pose x']), float(values[i]['marker pose y'])))
    #     if int(values[i]['num dirt low level']) >= 6000 and int(values[i]['num dirt low level']) < 10000 :
    #         marker_dict['Yellow'].append(get_pose(float(values[i]['marker pose x']), float(values[i]['marker pose y'])))

    #     if int(values[i]['num dirt low level']) >= 10000 :
    #         marker_dict['Red'].append(get_pose(float(values[i]['marker pose x']), float(values[i]['marker pose y'])))




    while not rospy.is_shutdown():
        
        # for key in marker_dict.keys():
        #     for i in range(len(marker_dict[key])):
        #         markers.id +=1
        #         markers.points = [marker_dict[key][i]]
        #         markers.color = get_color(key)
        #         markers.header.stamp = rospy.Time.now()    

                marker_pub.publish(markers)
                rospy.sleep(0.01)


