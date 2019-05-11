#!/usr/bin/env python
import os
import sys
import rospy
from rulo_base.msg import  Reference, Coordinates
from rulo_base.srv import  TfPose

if __name__ == '__main__':
    rospy.init_node('tf_client_node')
    rospy.wait_for_service('tf_service')
    ref = rospy.ServiceProxy('tf_service', TfPose)
    
    send = ref(Reference(*('source', 'target')))
    print send.pose.x

