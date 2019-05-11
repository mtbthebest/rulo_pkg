#!/usr/bin/env python
import os

from rulo_base.Rulo import Rulo as rulo
import numpy as np
from geometry_msgs.msg import Twist
import rospy
rospy.init_node('rotate', anonymous=True)

rulo().rotate_with_vel()