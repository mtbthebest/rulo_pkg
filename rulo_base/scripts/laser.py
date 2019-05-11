#!/usr/bin/env python
import os
import rospy
from sensor_msgs.msg import LaserScan
from rulo_msgs.msg import Range
def callback(msg):
    range_ahead = min(msg.ranges[0:len(msg.ranges)/2])
    print range_ahead
    # print msg.range

if __name__ == '__main__':
    rospy.init_node("laser", anonymous=True)
    scan_subs = rospy.Subscriber("/scan", LaserScan, callback)
    # scan = rospy.Subscriber("/mobile_base/event/optical_ranging_sensor", Range, callback)
    
    rospy.spin()
                     