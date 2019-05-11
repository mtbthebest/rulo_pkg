#!/usr/bin/env python
import os
import rospy
from geometry_msgs.msg import Twist, TwistStamped
class TwistedStamp:
    def __init__(self):
        rospy.init_node('twist')
        # rospy.on_shutdown(self.shutdown)
        self.rate = 10
        self.r = rospy.Rate(self.rate)
        self.cmd_pub = rospy.Publisher('/cmd_vel_stamp', TwistStamped, queue_size=5)
        self.cmd_vel_stamp = TwistStamped()

        while not rospy.is_shutdown():
            self.twist_stamp_sub = rospy.Subscriber('/Rulo/cmd_vel', Twist,self.callback)                
            self.cmd_vel_stamp.header.stamp = rospy.Time.now()

            if self.msg:
                self.cmd_vel_stamp.twist = self.msg
            else:
                
                self.cmd_vel_stamp.twist = Twist()
                                
            self.cmd_pub.publish(self.cmd_vel_stamp)
            rospy.sleep(0.1)
        
    def callback(self, msg):
         self.msg = msg
       
        
              
if __name__ == '__main__':
   TwistedStamp()
  
