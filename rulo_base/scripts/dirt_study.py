#!/usr/bin/env	python
import rospy
import message_filters
from geometry_msgs.msg import PoseStamped, Point
from rulo_msgs.msg import   BrushesPWM_cmd
from rulo_base.msg import BrushStamped
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TwistStamped
from rulo_msgs.msg import DirtDetect
from rulo_utils.csvcreater import csvcreater
from rulo_utils.csvwriter import csvwriter
from collections import OrderedDict

filename = '/home/ubuntu/catkin_ws/src/rulo_pkg/rulo_base/data/test2017-11-10_20h15.csv'
headers = ['p_x', 'p_y', 'p_z', 'q_x', 'q_y', 'q_z', 'q_w',
           'dirt_high_level', 'dirt_low_level', 'scan_ranges', 'wall_time']



class Dirt:
    def __init__(self):
        rospy.init_node('dirt_cond_prob')
        self.pose_sub = message_filters.Subscriber('/tf_pub', PoseStamped)
        self.dirt_level_sub = message_filters.Subscriber('/mobile_base/event/dirt_detect', DirtDetect)
        self.vel_sub = message_filters.Subscriber('/cmd_vel_stamp',TwistStamped )
        self.brush_pwm_sub = message_filters.Subscriber('/brush_stamp_pwm', BrushStamped)
        # self.laser = message_filters.Subscriber('/scan', LaserScan)
        

        self.indice =0
        # self.mf = message_filters.ApproximateTimeSynchronizer(
        #     [self.sub1, self.sub2, self.sub3], 10, 0.05)
        # self.mf.registerCallback(callback)
        # rospy.spin()
    
    def brush_pwm_cte_vel_var(self):
        print 'Brush: cte , Vel: variable'
        self.filename = '/home/ubuntu/catkin_ws/src/rulo_pkg/rulo_base/data/brush_pwm_cte_vel_var.csv'
        self.mf = message_filters.ApproximateTimeSynchronizer([self.brush_pwm_sub,
                                                              self.pose_sub, 
                                                              self.vel_sub,
                                                              self.dirt_level_sub], 10, 1.0)
        self.mf.registerCallback(self.callback)
        rospy.spin()       
    def callback(self, *args):
        #   self.wall_time = rospy.get_time()
        #   self.
        data = OrderedDict()
        for i in range(len(args)): 
            argument = args[i].__slots__
            if argument == ['header', 'brush']:
                data['main_brush']= [args[i].brush.main_brush]
                data['side_brush'] = [args[i].brush.side_brush]
                data['vacuum'] = [args[i].brush.vacuum]
            elif argument == ['header', 'twist']:
                 data['lin_x'] = [args[i].twist.linear.x]
                 data['rot_z'] = [args[i].twist.angular.z] 
            elif argument == ['header', 'dirt_high_level', 'dirt_low_level']:
                 data['dirt_high_level'] = [args[i].dirt_high_level]
                 data['dirt_low_level'] = [args[i].dirt_low_level]

            elif argument == ['header', 'pose']:
                 data['pose_x'] = [args[i].pose.position.x]
                 data['pose_y'] = [args[i].pose.position.y]
                 data['pose_z'] = [args[i].pose.position.z]
                 data['orientation_x'] = [args[i].pose.orientation.x]
                 data['orientation_y'] = [args[i].pose.orientation.y]
                 data['orientation_z'] = [args[i].pose.orientation.z]
                 data['orientation_w'] = [args[i].pose.orientation.w]
            
                

        csvwriter(self.filename, headers= data.keys(), rows = data.values())
                
    


if __name__ == '__main__':
    Dirt().brush_pwm_cte_vel_var()

        
    
