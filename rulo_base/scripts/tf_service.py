#!/usr/bin/env python
import os
import sys
import rospy
from rulo_base.msg import  Reference, Coordinates
from rulo_base.srv import  TfPose, TfPoseResponse


def callback(request):
    b= Coordinates()
    b.x = 2.0
    b.y = 4.0
    b.theta = 1.5

    return TfPoseResponse(b)

if __name__ =='__main__':
    rospy.init_node('tf_service_node')
    service = rospy.Service('tf_service', TfPose, callback)
    rospy.spin()