#!/usr/bin/env	python
import rospy
import message_filters
from geometry_msgs.msg import PoseStamped, Point
from sensor_msgs.msg import LaserScan
from rulo_msgs.msg import DirtDetect
from rulo_utils.csvcreater import csvcreater
from rulo_utils.csvwriter import csvwriter
filename = '/home/ubuntu/catkin_ws/src/rulo_pkg/rulo_base/data/test2017-11-10_20h15.csv'
headers = ['p_x', 'p_y', 'p_z', 'q_x', 'q_y', 'q_z', 'q_w',
           'dirt_high_level', 'dirt_low_level', 'scan_ranges', 'wall_time']



class GetData:
    def __init__(self):
        rospy.init_node('data')
        self.sub1 = message_filters.Subscriber('tf_pub', PoseStamped)
        self.sub2 = message_filters.Subscriber('/mobile_base/event/dirt_detect', DirtDetect)
        self.sub3 = message_filters.Subscriber('/scan', LaserScan)
        self.indice =0
        self.mf = message_filters.ApproximateTimeSynchronizer(
            [self.sub1, self.sub2, self.sub3], 10, 0.05)
        self.mf.registerCallback(callback)
        rospy.spin()
        
    def callback(self, sub1, sub2, sub3):
        
  
        self.wall_time = rospy.get_time()
        self.p_x, self.p_y, self.p_z = self.sub1.pose.position.x, self.sub1.pose.position.y, self.sub1.pose.position.z
        self.q_x, self.q_y, self.q_z, self.q_w = self.sub1.pose.orientation.x, self.sub1.pose.orientation.y, self.sub1.pose.orientation.z, self.sub1.pose.orientation.w
        self.dirt_h_lev = self.sub2.dirt_high_level
        self.dirt_l_lev = self.sub2.dirt_low_level
        self.scan_ranges = self.sub3.ranges

        rows = [[self.p_x], [self.p_y], [self.p_z], [self.q_x], self.q_y], [self.q_z], [self.q_w], [
            self.dirt_h_lev], [self.dirt_l_lev], [self.scan_ranges], [self.wall_time]]

        csvwriter(filename, headers, rows)
        self.indice +=1
        print self.indice


if __name__ == '__main__':
    GetData()

        
    
            #!/usr/bin/env	python
            # import rospy
            # import message_filters
            # from geometry_msgs.msg import PoseStamped, Point
            # from sensor_msgs.msg import LaserScan
            # from rulo_msgs.msg import DirtDetect
            # from rulo_utils.csvcreater import csvcreater
            # from rulo_utils.csvwriter import csvwriter
            # filename = '/home/ubuntu/catkin_ws/src/rulo_pkg/rulo_base/data/test2017-11-10_20h15.csv'
            # headers = ['p_x', 'p_y', 'p_z', 'q_x', 'q_y', 'q_z', 'q_w', 'dirt_high_level', 'dirt_low_level', 'scan_ranges', 'wall_time']

            # def callback(sub1, sub2, sub3):

            # wall_time= rospy.get_time()
            # p_x, p_y, p_z = sub1.pose.position.x, sub1.pose.position.y, sub1.pose.position.z
            # q_x, q_y, q_z, q_w= sub1.pose.orientation.x, sub1.pose.orientation.y, sub1.pose.orientation.z, sub1.pose.orientation.w
            # dirt_h_lev= sub2.dirt_high_level
            # dirt_l_lev = sub2.dirt_low_level
            # scan_ranges = sub3.ranges

            # rows = [[p_x], [p_y], [p_z], [q_x], [q_y], [q_z], [q_w], [dirt_h_lev], [dirt_l_lev], [scan_ranges], [wall_time]]

            # csvwriter(filename, headers, rows)



            # rospy.init_node('dirt_map')
            # start= rospy.get_time()

            # sub1 =	message_filters.Subscriber('tf_pub', PoseStamped)
            # sub2=	message_filters.Subscriber('/mobile_base/event/dirt_detect', DirtDetect)
            # sub3= message_filters.Subscriber('/scan', LaserScan)
            # mf = message_filters.ApproximateTimeSynchronizer([sub1, sub2, sub3], 10, 0.05)
            # mf.registerCallback(callback)
            # rospy.spin()
