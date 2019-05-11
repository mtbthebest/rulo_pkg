#!/usr/bin/env	python
import rospy
import message_filters
from geometry_msgs.msg import PoseStamped, Point
from rulo_msgs.msg import DirtDetect
from rulo_utils.csvwriter import csvwriter
from collections import OrderedDict


filename = '/home/ubuntu/catkin_ws/src/rulo_pkg/rulo_base/data/pwm_40_lin_0.03.csv'
headers = ['dirt_high_level', 'dirt_low_level', 'wall_time']


class Dirt:
    def __init__(self):
        rospy.init_node('dirt_cond_prob')    
    def get_dirt(self):
        rospy.loginfo('Subscribing to dirt topic')
        self.dirt_level_sub = rospy.Subscriber(
            '/mobile_base/event/dirt_detect', DirtDetect, self.callback)
        rospy.spin()

    def callback(self, msg):
        self.wall_time = rospy.get_time()
        print msg.dirt_high_level, msg.dirt_low_level, self.wall_time
        csvwriter(self.filename, headers=headers, rows=[[msg.dirt_high_level], [msg.dirt_low_level], [self.wall_time]]

    def get_rot_dirt(self):
        self.time=rospy.Time.now()
        while rospy.Time.now() - self.time < rospy.Duration(60):
            self.dirt_level_sub=rospy.Subscriber(
                '/mobile_base/event/dirt_detect', DirtDetect, self.callback)
            rospy.sleep(0.02)
        break

        
      
if __name__ == '__main__':
    Dirt().get_rot_dirt()
