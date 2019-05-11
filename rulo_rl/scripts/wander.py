#!/usr/bin/env	python
import rospy
import os
from actionlib_msgs.msg import *
import actionlib
from geometry_msgs.msg import Point, Quaternion, Pose, PoseWithCovarianceStamped, PoseWithCovariance
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from tf.transformations import quaternion_from_euler
from rulo_utils.csvreader import csvread
from rulo_utils.csvwriter import csvwriter
 
pose_filename = '/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/data/pose.csv'
time_origin = '/home/mtb/rulo_ws/src/rulo_pkg/rulo_rl/data/origin.csv'
time_cells =    '/home/mtb/rulo_ws/src/rulo_pkg/rulo_rl/data/time.csv'
index_filename = '/home/mtb/rulo_ws/src/rulo_pkg/rulo_rl/data/index.csv'
namespace = 'screen2/'
class Wander:
    def __init__(self):
        rospy.init_node('wander')
        self.r = rospy.Rate(10)
        self.goal = MoveBaseGoal()
        self.get_data()
        self.move_base = actionlib.SimpleActionClient(namespace+"move_base", MoveBaseAction)
        self.move_base.wait_for_server(rospy.Duration(10))
        self.goal = MoveBaseGoal()
        self.goal.target_pose.header.frame_id = 'map'
        
    def navigate(self, coordinates):
            trans = Point(*(coordinates[0], coordinates[1], 0.0))
            rot = Quaternion(*(quaternion_from_euler(0.0, 0.0, 0.0, axes='sxyz')))
            pose = Pose(*(trans, rot))
            self.goal.target_pose.pose = pose
            print 'initial_pose: ', pose
            self.goal.target_pose.header.stamp = rospy.Time.now()
            self.move_base.send_goal(self.goal)
            finished_within_time = self.move_base.wait_for_result(
                rospy.Duration(300.0))
            finish = 'Nan'
            if not finished_within_time:
                self.move_base.cancel_goal()                 
            else:
                state = self.move_base.get_state()
                if state == GoalStatus.SUCCEEDED:
                    finish = rospy.get_time()
                    rospy.loginfo('Succeeded')   
          
            return finish
    
    def wander(self):
        while not rospy.is_shutdown():
            indice_i  = 300#, indice_j = 300,301#self.restore()               
            for i in range(indice_i, len(self.position)):
                print 'x: ', str(i) , "  ", self.position[i]
                self.initial_pose(pose=[0.0, 0.0])
                start = rospy.get_time()
                finish = self.navigate(self.position[i])
                csvwriter(time_origin,
                          headers=['pose', 'start', 'finish'],
                          rows = [[i], [start], [finish]])
                rospy.sleep(2.0)     
                while not rospy.is_shutdown():
                    # if indice_j > i+1:
                    #     goal = indice_j
                    # else:
                    goal = i+1
                    for j in range(goal, len(self.position)):
                            self.initial_pose(self.position[i])
                            print [i, j]
                            start = rospy.get_time()
                            finish = self.navigate(self.position[j])                            
                            csvwriter(time_cells,
                                    headers=['pose', 'start', 'finish'],
                                    rows=[[[i,j]], [start], [finish]])
                            rospy.sleep(2.0)
                            csvwriter(index_filename, ['i', 'j'], [[i], [j]])
                    # indice_j = 0
                    break
            break
        
        print 'Finished wandering'
                            
                        
                        

        
    def get_data(self):
        data = csvread(pose_filename)
        self.position = []
        for str_pose in data['pose']:
            pose = []
            for elem in str_pose[1:-1].split(','):
                pose.append(float(elem))
            self.position.append(pose)
    
    def initial_pose(self, pose):
        self.initial_pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=10)
        self.pose_cov = PoseWithCovarianceStamped()
        self.pose_cov.header.frame_id = 'map'
        
        trans = Point(*(pose[0], pose[1], 0.0))
        rot = Quaternion(*(quaternion_from_euler(0.0,0.0,0.0, axes='sxyz')))
        covariance = [0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06853891945200942]

        pose = PoseWithCovariance(*(Pose(*(trans, rot)), covariance))
        self.pose_cov.header.seq = rospy.Time.now()
        self.pose_cov.pose = pose
        rospy.loginfo('initial pose: ' + str(pose))
        k = 0
        while not rospy.is_shutdown():
            # while not rospy.Time.now() - self.pose_cov.header.seq < rospy.Duration(10.0):
                self.initial_pub.publish(self.pose_cov)
                rospy.sleep(2)
                k +=1
                if k>2:
                     break
        # self.initial_pub.publish(self.pose_cov)
        # self.r.sleep()
        # self.initial_pub.publish(self.pose_cov)
        # self.r.sleep()
        # self.initial_pub.publish(self.pose_cov)
        # self.r.sleep()

    def restore(self):
        if not os.path.isfile(index_filename): return 0,1
        else:
            data = csvread(index_filename)
            indice_i = int(data['i'][-1])
            indice_j = int(data['i'][-1])
            return indice_i, indice_j

        

            


if __name__ == '__main__':
    Wander().navigate([1.0,-2.0])


