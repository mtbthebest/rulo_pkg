#!/usr/bin/env python
import os
import rospy
from rulo_base.msg import BrushStamped
from rulo_msgs.msg import BrushesPWM_cmd
import dynamic_reconfigure.server
from rulo_base.cfg import BrushPWMConfig

class BrushStamp:
    def __init__(self):
        rospy.init_node('brush_stamp')
       
        self.rate = 20
        self.r = rospy.Rate(self.rate)
        self.brush_pwm_sub = rospy.Subscriber('/mobile_base/command/brushesPWM_cmd', BrushesPWM_cmd,self.callback)
        rospy.spin()
     
    def callback(self, msg):
          self.brushes_pub = rospy.Publisher('/brush_stamp_pwm', BrushStamped, queue_size=5)
          self.brushes_stamp = BrushStamped()
          self.brushes_stamp.header.stamp = rospy.Time.now()
          self.brushes_stamp.brush.main_brush = msg.main_brush
          self.brushes_stamp.brush.side_brush = msg.side_brush
          self.brushes_stamp.brush.vacuum = msg.vacuum     
          self.brushes_pub.publish(self.brushes_stamp)
  
       

if __name__ == '__main__':
    BrushStamp()
  
