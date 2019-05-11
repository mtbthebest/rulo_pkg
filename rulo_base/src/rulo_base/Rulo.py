#!/usr/bin/env python


import rospy
import actionlib
from actionlib_msgs.msg import *
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped, Point, Quaternion, Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from random import sample
from math import pow, sqrt
import tf
from tf.transformations import euler_from_quaternion , quaternion_from_euler

target_frame = 'map'
source_frame = 'base_footprint'


class Rulo():
    def __init__(self):
        self.rate = 20
        self.r = rospy.Rate(self.rate)
        self.cmd_vel_pub = rospy.Publisher('/Rulo/cmd_vel', Twist, queue_size=10)
        self.move_cmd = Twist()
        self.move_cmd.linear.y = 0
        self.move_cmd.linear.z = 0
        self.move_cmd.angular.x = 0
        self.move_cmd.angular.y = 0
        
        # self.tf_listener = tf.TransformListener()    
        # rospy.sleep(1.0)
        # try: 
        #     self.tf_listener.waitForTransform(target_frame, source_frame, rospy.Time(), rospy.Duration(1.0))
        # except (tf.Exception, tf.ConnectivityException, tf.LookupException):
        #     rospy.logerr('tf exception')
    
        self.goal_states = ['PENDING', 'ACTIVE', 'PREEMPTED',
                            'SUCCEEDED', 'ABORTED', 'REJECTED',
                            'PREEMPTING', 'RECALLING', 'RECALLED',
                            'LOST']
        self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)       
        self.move_base.wait_for_server(rospy.Duration(10))
        rospy.loginfo("Connected to move base server")
        self.goal = MoveBaseGoal()
        self.goal.target_pose.header.frame_id = '/map'

    def nav_tf(self, *target):

            self.goal.target_pose.pose.position.x = target[0][0]
            self.goal.target_pose.pose.position.y = target[0][1]
            self.goal.target_pose.pose.position.z = 0
            self.goal.target_pose.pose.orientation = Quaternion(*(quaternion_from_euler(0, 0 , target[0][2], axes='sxyz')))
            self.goal.target_pose.header.stamp = rospy.Time.now()
            self.move_base.send_goal(self.goal)

            finished_within_time = self.move_base.wait_for_result(rospy.Duration(60))

            if not finished_within_time:
                self.move_base.cancel_goal()
                rospy.loginfo("Timed out achieving goal")
            else:
                state = self.move_base.get_state()

                if state == GoalStatus.SUCCEEDED:
                    rospy.loginfo("Goal succeeded!")
                    rospy.loginfo("State:" + str(state))
                else:
                  rospy.loginfo("Goal failed with error code: " + str(self.goal_states[state]))            

    def rotate_with_vel(self, vel_angular=0.3, duration=None):
        self.move_cmd.linear.x = 0
        self.move_cmd.angular.z = vel_angular

        if duration:
            start = rospy.Time.now()
            while not rospy.is_shutdown():
                while ((rospy.Time.now() - start) < rospy.Duration(duration)):                 
                        self.move()
                self.shutdown()
        else:
            while not rospy.is_shutdown():
                self.move()

    def go_with_vel(self, vel_linear=0.5, duration=None):
        self.move_cmd.linear.x = vel_linear
        self.move_cmd.angular.z = 0

        if duration:
            start = rospy.Time.now()
            while not rospy.is_shutdown():
                while ((rospy.Time.now() - start) < rospy.Duration(duration)):                 
                        self.move()
                self.shutdown()
        else:
            while not rospy.is_shutdown():
                self.move()

    def move (self):
        self.cmd_vel_pub.publish(self.move_cmd)
        self.r.sleep()
    
    def get_tf(self, target_frame ='map', source_frame='base_footprint'):             
        while not rospy.is_shutdown():    
            (trans, rot)= self.tf_listener.lookupTransform(target_frame, source_frame, rospy.Time()) 
            print(trans)
             
    def turn_angle(self, target):
        pass

    def convert_angle(self, angle):
        try:
            a =pow((0.708218042904 /0.9999999998175838), 90/angle)  * (0.9999999998175838)
      
        except ZeroDivisionError:
            a = 0
        
        print a
        return a
        
    def dirt_detect(self):
        pass
    
    def shutdown(self):
        rospy.signal_shutdown('Over')


    # def get_odom(self):
    #     (trans,rot) = self.tf_listener.lookupTransform('/map', '/base_footprint')
    #     return euler_from_quaternion(Quaternion(*rot))

    # def shutdown(self):
    #     rospy.loginfo("Stopping the robot...")
    #     self.move_base.cancel_goal()
    #     rospy.sleep(2)
    #     self.cmd_vel_pub.publish(Twist())
    #     rospy.sleep(1)


# if __name__ == '__main__':
#     try:
#         Rulo().nav_tf(1,1,0)
#     except rospy.ROSInterruptException:
#         rospy.loginfo("Navigation test finished.")
