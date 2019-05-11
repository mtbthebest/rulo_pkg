#!/usr/bin/env	python
import	rospy
from	std_msgs.msg	import	String
from	geometry_msgs.msg	import	Twist
import dynamic_reconfigure.server 
from rulo_base.cfg import VelocityConfig
trans_vel = 0.04
rot_vel = 0.25




class Teleop:
    def __init__(self):
    		rospy.init_node('teleop')
	    	self.trans_vel = rospy.get_param('linear_x', trans_vel)
		self.rot_vel = rospy.get_param('angular_z', rot_vel)		
		self.twist_pub = rospy.Publisher('/Rulo/cmd_vel', Twist, queue_size=10)
		self.twist = Twist()
		self.dyn_server = dynamic_reconfigure.server.Server(VelocityConfig,
															self.dynamic_reconfigure_callback)
		rospy.Subscriber('keys',String,	self.keys_cb)
		rospy.spin()
		
    def keys_cb(self, msg):
	    			self.key_mapping = {	'\x41':	[	0,	self.trans_vel],
                           '\x42':	[0,	-self.trans_vel],
                           '\x44':	[self.rot_vel,	0],
                           '\x43':	[-self.rot_vel,		0],
                           '\x20':	[	0,	0]	}
				if	len(msg.data)	==	0	or	not	self.key_mapping.has_key(msg.data[0]):
						return  # unknown	key
				
				# print msg
				
				self.vels	=	self.key_mapping[msg.data[0]]
				
				
				self.twist.angular.z	=	self.vels[0]
				self.twist.linear.x		=	self.vels[1]
				
				self.twist_pub.publish(self.twist)
    def dynamic_reconfigure_callback(self,config, level):
        if self.rot_vel != config['angular_z']:
            self.rot_vel = config['angular_z']
        if self.trans_vel != config['linear_x']:
            self.trans_vel = config['linear_x']     
        return config
if	__name__	==	'__main__':
    	Teleop()
		
