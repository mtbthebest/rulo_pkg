#!/usr/bin/env python
import sys
import rospy
import numpy as np
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Header
from tf import TransformListener
import math
from visualization_msgs.msg import Marker


count = 0
MARKERS_MAX = 100

class robotGame():
    def __init__(self):
        #init code
        rospy.init_node("robotGame")
        self.currentDist = 1
        self.previousDist = 1
        self.reached = False
        self.tf = TransformListener()

        self.joint_names =['r_wheel_joint', 'l_wheel_joint', 'torso_lift_joint', 'head_pan_joint', 'head_tilt_joint', 'shoulder_pan_joint', 'shoulder_lift_joint', 'upperarm_roll_joint', 'elbow_flex_joint', 'forearm_roll_joint', 'wrist_flex_joint', 'wrist_roll_joint', 'r_gripper_finger_joint', 'l_gripper_finger_joint', 'bellows_joint']
        self.joint_positions =[0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.1, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]      
        self.jv = []
    
        self.rate =100
        self.r = rospy.Rate(self.rate)
        self.pub = rospy.Publisher('joint_states', JointState, queue_size=10) 
        self.markers= rospy.Publisher('vis',Marker, queue_size=10)
        self.js = JointState()
        self.js.header = Header()
        self.js.name = self.joint_names
        self.js.velocity = []
        self.js.effort = []

      

        # initial starting location I might want to move to the param list
    
        self.marker = Marker()
        self.marker.header.frame_id = "/base_link"
        self.marker.id =1
        self.marker.type = self.marker.SPHERE
        self.marker.action = self.marker.ADD
        self.marker.scale.x = 0.2
        self.marker.scale.y = 0.2
        self.marker.scale.z = 0.2
        self.marker.color.a = 1.0
        self.marker.color.r = 1.0
        self.marker.color.g = 1.0
        self.marker.color.b = 0.0
        self.marker.pose.orientation.w = 1.0
        self.marker.pose.position.x = math.cos(count / 50.0)
        self.marker.pose.position.y = math.cos(count / 40.0) 
        self.marker.pose.position.z = math.cos(count / 30.0) 
        self.marker.lifetime = rospy.Duration(2)

        

        
        start = rospy.Time.now()
        self.sub_list = list()
        self.current_joint_values = list()
        self.move_reset(2,  self.joint_positions )
        
            
        self.jv =  self.joint_positions
        # print self.jv
        self.destPos = np.random.uniform(0,0.25, size =(3))
       
        #self.reset()

#     def jsCB(self,msg):
#         temp_dict = dict(zip(msg.name, msg.position))
#         self.sub_list.append(temp_dict)
#         # self.jv = [temp_dict[x] for x in self.joint_names]       
#         # self.js.position = self.joint_position
#         # print msg.position
   
      
    def getCurrentJointValues(self):
        return self.jv[-1]

    def getCurrentPose(self):
        self.tf.waitForTransform("/base_link","/r_gripper_finger_link",rospy.Time(),rospy.Duration(10))
        t = self.tf.getLatestCommonTime("/base_link", "/r_gripper_finger_link") 
        position, quaternion = self.tf.lookupTransform("/base_link","/r_gripper_finger_link",t)
        return [position[0],position[1],position[2]]

    def setJointValues(self,tjv):
        self.joint_positions = np.zeros(14).tolist()
        self.joint_positions[5:11] = tjv  
        rospy.sleep(0.20)
    
        return self.joint_positions

    def getDist(self, positions,goal):
        position = np.array(positions)
        goal = np.array(goal)
        return np.linalg.norm(position - goal)

    def reset(self):
        # print self.jv
        self.destPos = np.random.uniform(0,1, size =(3))
        self.jv.append(self.setJointValues(np.random.uniform(-1,1, size=(7)).tolist()))
        
        self.move_target(2, self.jv[-1],self.destPos )
    
        tjv = self.getCurrentJointValues()
        positions = self.actual_jv[100]
     
        # positions = self.getCurrentPose()
        # # print positions
        # # print self.destPos.tolist()
        # print('tjv' , tjv,'pos', positions, self.destPos.tolist())       
        #print tjv
        # print self.jv
        return  (tjv+positions+self.destPos.tolist())
        
       
#         # return tjv+positions+self.destPos.tolist()
 
    def step(self,vals, goal):
        done = False
      
        jv = self.jv[-1][5:12] 
        # print jv
        # print ('vals '+ str(vals.flatten().tolist()))
        tjv_m = [x + y for x,y in zip(vals.flatten().tolist(), jv)]
        tjv = self.setJointValues(tjv_m)
        self.jv.append(tjv)
        self.move_target(2,tjv,goal)
        positions = self.actual_jv[100]   
       
        curDist = self.getDist(goal, positions)
        
        reward = -curDist - 0.00*np.linalg.norm(vals) - 0.5*math.log10(curDist) 
        #print  goal, -curDist - 0.5*math.log10(curDist) ,-curDist, np.linalg.norm(vals)
        if curDist < 0.2:
            reward +=10 
            done = True
        tjv = self.getCurrentJointValues()
        goals = list()

        for i in range(len(goal)):
            goals.append(goal[i])
        
        # print ('reward', reward)
        return [tjv+positions+goals, reward, done]

    def done(self):
        #self.sub.unregister()
        rospy.signal_shutdown("done")

    def move_reset(self, duration, value):
         start = rospy.Time.now()
        
         while(rospy.Time.now() - start < rospy.Duration(duration)):
           
                self.js.header.stamp = rospy.Time.now()
                self.js.position = value
                self.pub.publish(self.js)
                self.markers.publish(self.marker) 
                #self.sub = rospy.Subscriber('joint_states', JointState, self.jsCB)
            
                self.r.sleep()
         


    def move_target(self, duration, value,dest ):
         start = rospy.Time.now()
         self.js.position = value
         self.actual_jv = list()
         self.marker.pose.position.x = dest[0]
         self.marker.pose.position.y = dest[1] 
         self.marker.pose.position.z = dest[2]
         while(rospy.Time.now() - start < rospy.Duration(duration)):
           
                self.js.header.stamp = rospy.Time.now()
                self.actual_jv.append(self.getCurrentPose())
                self.pub.publish(self.js)
                self.markers.publish(self.marker)
                # print self.actual_jv
                
                            
                self.r.sleep()
        
# if __name__ == "__main__":
#             r = robotGame()
#             # print r.getCurrentJointValues()
#             # print r.getCurrentPose()
#             # r.reset()
#             # print r.getCurrentJointValues()
#             # print r.getCurrentPose()




