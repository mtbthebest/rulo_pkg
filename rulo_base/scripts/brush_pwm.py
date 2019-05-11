#!/usr/bin/env python
import os
import rospy
# from rulo_base.msg import BrushStamped
from rulo_msgs.msg import   BrushesPWM_cmd
import dynamic_reconfigure.server
from rulo_base.cfg import BrushPWMConfig

class Brush:
    def __init__(self):
        rospy.init_node('brush')
        rospy.on_shutdown(self.shutdown)
        self.rate = 10
        self.r = rospy.Rate(self.rate)
        self.brushes_pub = rospy.Publisher('/mobile_base/command/brushesPWM_cmd', BrushesPWM_cmd,queue_size=5)
        self.pwm = 40
        self.brush_pwm= BrushesPWM_cmd()

        self.brush_pwm.main_brush = rospy.get_param('main_brush',self.pwm)
        self.brush_pwm.side_brush = rospy.get_param('side_brush', self.pwm)
        self.brush_pwm.vacuum = rospy.get_param('vacuum', self.pwm)
        self.dyn_server = dynamic_reconfigure.server.Server(BrushPWMConfig, self.dynamic_reconfigure_callback)
        while not rospy.is_shutdown():
            self.brushes_pub.publish(self.brush_pwm)            
            self.r.sleep()

        
    def dynamic_reconfigure_callback(self,config, level):
        if self.brush_pwm.main_brush != config['main_brush']:
            self.brush_pwm.main_brush = config['main_brush']
        if self.brush_pwm.side_brush != config['side_brush']:
            self.brush_pwm.side_brush = config['side_brush']
        if self.brush_pwm.vacuum!= config['vacuum']:
            self.brush_pwm.vacuum = config['vacuum']         
        return config

    def shutdown(self):
        rospy.loginfo('Shutting down')
        self.brushes_pub.publish(BrushesPWM_cmd())


if __name__ == '__main__':
    Brush()
  
