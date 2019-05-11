#!/usr/bin/env python
import os
import rospy
from rulo_msgs.msg import   BrushesPWM_cmd, DirtDetect
from std_msgs.msg import Header
from geometry_msgs.msg import Point, Quaternion, Pose
import tf
from tf.listener import TransformListener
import csv
from collections import OrderedDict

target_frame = '/map'
source_frame = '/base_footprint'


class DirtCount:
    def __init__(self):

        rospy.init_node('dirt')
        self.tf_listener = TransformListener()
        rospy.sleep(2.0)
        self.tf_pub = rospy.Publisher("tf_pub", Point, queue_size=10)
        try: 
            self.tf_listener.waitForTransform(target_frame, source_frame, rospy.Time(), rospy.Duration(1.0))
            rospy.loginfo('TF server done')
        except (tf.Exception, tf.ConnectivityException, tf.LookupException):
            rospy.signal_shutdown('tf exception')
        
        self.counter=dict()
        self.count=dict()
        self.counter_list = list()
        self.dirt_dict = OrderedDict()
        self.timer = OrderedDict()
        self.level_dict = dict()
        self.dirt_list = list()
        self.counter['low']= []
        self.counter['high']= []
        self.count['low']= 0
        self.count['high']= 0
        self.dirt_dict['low']= []
        self.dirt_dict['high']= []
        self.timer['low']= []
        self.timer['high']= []
      
        self.reset_status = False
        
        
        self.tf = OrderedDict()
        self.tf['x'] = []
        self.tf['y'] = []
        self.mark_pos = OrderedDict()
        self.mark_pos['x_min'] =[0]
        self.mark_pos['x_max'] =[0]
        self.mark_pos['y_min'] =[0]
        self.mark_pos['y_max'] =[0]
        self.start = rospy.Time.now()
        self.start_timer = rospy.get_time()

        

      
        # rospy.Subscriber('tf_pub', Point, self.tf_call)
        rospy.Subscriber('/mobile_base/event/dirt_detect', DirtDetect, self.dirt_call)
        rospy.on_shutdown(self.shutdown)
        rospy.loginfo("start")
        rospy.spin()
        rospy.loginfo("end")       
    
  

    def dirt_call(self, msg):
        self.trans, self.rot = self.tf_listener.lookupTransform(target_frame, source_frame, rospy.Time())
        self.tf['x'].append(self.trans[0])
        self.tf['y'].append(self.trans[1])
        max_low= msg.dirt_low_level 
        max_high = msg.dirt_high_level 
        # rospy.loginfo('max: ' + str(max_low))     
       
        if (len(self.dirt_list)):
            check_low = self.check_reset('low',max_low)
            if check_low:
                self.update('low',max_low)
                self.update_marker(self.tf)
                rospy.loginfo(self.mark_pos['x_max'])
                del self.tf['x'][:]
                del self.tf['y'][:]

            check_high = self.check_reset('high',max_high)
            if check_high:
                self.update('high',max_high)
        
        self.level_dict['low'] =msg.dirt_low_level 
        self.level_dict['high'] =msg.dirt_high_level 
        self.dirt_list.append(self.level_dict)
        # rospy.loginfo(self.dirt_list)        
        # if (rospy.Time.now() - self.start > rospy.Duration(60) ):
        #     self.shutdown()


    def update_marker(self,tf_):
        self.mark_pos['x_min'].append(min(tf_['x']))
        self.mark_pos['x_max'].append(max(tf_['x']))
        self.mark_pos['y_min'].append(min(tf_['y']))
        self.mark_pos['y_max'].append(max(tf_['y']))

    def check_reset(self, key, value):       
        for j in range(len(self.dirt_list)):
                if (value >= self.dirt_list[j][key]):
                        continue
                else:
                    self.reset_status = True
                    break
        return self.reset_status 
    

    def update(self,key,value):
            if not key in self.count.keys():
                self.count[key] =0
                self.counter[key] =[]
            self.count[key] +=1
            print ('counter update ' + str(self.count))
            self.counter[key].append(int(256 - self.dirt_list[-1][key] +value))
            self.timer[key].append(rospy.get_time() - self.start_timer)
            self.dirt_dict[key].append(self.counter[key][-1])
            self.reset_status = False
            del self.dirt_list[:]
    
    def csv_dict(self, key, dic, field):
        if key:
            for i in range(1,len(dic[key])):
                    self.writer.writerow({field: dic[key][i]})
        else:
                for i in range(1,len(dic)):
                    self.writer.writerow({field: dic[i]})
    def shutdown(self):
        rospy.loginfo('Shutting down')
        num_low_level =  self.count['low']
        num_high_level = self.count['high']
        if self.timer['high']:
            timer_high = self.timer['high'][0]
        if not self.timer['high']:
            timer_high = None
        if self.timer['low']:
            timer_low= self.timer['low'][0]
        if not self.timer['low']:
            timer_low = None
        if self.dirt_dict['low']:
            dirt_low= self.dirt_dict['low'][0]
        if not self.dirt_dict['low']:
            dirt_low= 0
        if self.dirt_dict['high']:
            dirt_high= self.dirt_dict['high'][0]
        if not self.dirt_dict['high']:
            dirt_high= 0
        with open('/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/scripts/values.csv',  'w') as self.csvfile:
            self.fieldnames = ['count low level', 'count high level','time_low_level','num dirt low level', 'time_high_level', 'num dirt high level','marker pose min x','marker pose max x','marker pose min y','marker pose max y']
            self.writer = csv.DictWriter(self.csvfile, fieldnames=self.fieldnames)
            self.writer.writeheader()
      
            self.writer.writerow({'count low level': num_low_level, 'count high level': num_high_level,'time_low_level': timer_low, 'num dirt low level': dirt_low,'time_high_level': timer_high ,  'num dirt high level': dirt_high,'marker pose min x': self.mark_pos['x_min'][0],'marker pose max x' : self.mark_pos['x_max'][0],'marker pose min y': self.mark_pos['y_min'][0],'marker pose max y': self.mark_pos['y_max'][0] })
            self.csv_dict('low', self.timer,'time_low_level')
            self.csv_dict('low', self.dirt_dict,'num dirt low level')
            self.csv_dict('high', self.timer,'time_high_level')
            self.csv_dict('high', self.dirt_dict,'num dirt high level')
            self.csv_dict(key =None, dic=self.mark_pos['x_min'] , field=  'marker pose min x')
            self.csv_dict(key =None, dic=self.mark_pos['x_max'] , field=  'marker pose max x')
            self.csv_dict(key =None, dic=self.mark_pos['y_min'] , field=  'marker pose min y')
            self.csv_dict(key =None, dic=self.mark_pos['y_max'] , field=  'marker pose max y')

            # 'marker pose min x': self.mark_pos['x_min'][0],'marker pose max x' : self.mark_pos['x_max'][0],
            # 'marker pose min y': self.mark_pos['y_min'][0],'marker pose max y': self.mark_pos['y_max'][0]
         
        rospy.signal_shutdown('over')  
      

if __name__=="__main__":
    DirtCount()