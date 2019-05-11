#!/usr/bin/env	python
import os
import rospy
from actionlib_msgs.msg import *
import actionlib
from geometry_msgs.msg import Point, Quaternion, Pose, PoseWithCovarianceStamped, PoseWithCovariance,Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from tf.transformations import quaternion_from_euler
import sys
import numpy as np
import pandas as pd
from rulo_utils.graph_plot import Plot
from rulo_utils.csvwriter import csvwriter
from pandas import Series, DataFrame
from collections import OrderedDict
from rulo_base.markers import VizualMark,Line


class Path:
    
    def __init__(self):
        rospy.init_node('path')
        self.r = rospy.Rate(10)
        # self.cmd_pub = rospy.Publisher('/cmd_vel',Twist, queue_size=5)
        self.cmd_pub = rospy.Publisher('/Rulo/cmd_vel', Twist, queue_size=5)
        self.twist = Twist()
        self.goal = MoveBaseGoal()
        self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        self.move_base.wait_for_server(rospy.Duration(10))
        self.goal.target_pose.header.frame_id = 'map'
        self.pose = list(np.load('/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/data/pose.npy'))
        self.indice = np.load('/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/data/indices.npy')
        self.last_pose = []
    
    def visualize_dirt(self,filename):
        data = pd.read_csv(filename)
        self.dirt = [list(data['dirt_level'].values)[i] for i in self.indice]  
        thresh = 5000#max(self.dirt)
         
        color = [int(255 - float(self.dirt[i]) / thresh * 245) if self.dirt[i] <thresh else 10
                        for i in range(len(self.dirt))  ]  
        
        VizualMark().publish_marker(self.pose, sizes=[[0.25,0.25,0.0]] * 623 , color= color, 
                        convert = True, texture='Red',action='add', id_list = range(623),publish_num = 2)

        rospy.sleep(2)
    
    def clean_cells(self,cells=[],starts=0):
        # time_file = '/home/mtb/rulo_ws/src/rulo_pkg/rulo_ml/data/cells_time.csv'
        # pose =[]
        # time_cells = pd.read_csv(time_file)
        
        # self.navigate(self.pose[cells[0]])

        # # while :
           
        # for t in (cells):           
        #     VizualMark().publish_marker([self.pose[t]], sizes=[[0.25,0.25,0.0]] * 1 , color= ['Default'], 
        #                 action='delete', id_list = [t],publish_num = 10)
        #     if t ==0:
        #         duration =10.0
        #     else:
        #         duration += time_cells[str(cells[t-1])][cells[t]]
        #     if duration > 7200.0:
        #         break
        #     print t

       
      
        # for cell in cells:
        #     VizualMark().publish_marker([self.pose[cell]], sizes=[[0.25,0.25,0.0]] * 1 , color= ['Default'], action='delete', id_list = [cell],publish_num = 10) 
        #                     action='delete', id_list = [cell],publish_num = 10)
        # print start


        # print cells
        duration = 0.0
        start = rospy.get_time()
        last_time = start
        res = np.array(cells)[starts:-1:1]
        # print list(res)
        try:
            for cell in res:
                success =False
                if self.navigate(self.pose[cell]):
                    self.action_clean()
                    success = True  
                else:
                    pass
                duration +=rospy.get_time() - last_time
                last_time = rospy.get_time()
                if duration >=1800.0:
                    rospy.signal_shutdown()
                    break
                print cell , success
            
                csvwriter('/home/mtb/planning_high_04.csv',headers=['cells','results','duration'],rows=[[cell],[success],[duration]])
        except KeyboardInterrupt:
            rospy.signal_shutdown()
        
        self.navigate(self.pose[cells[0]])
        #         /*******
        
        #     print cell , ' ' , duration 
        # self.navigate(coordinates=[0.0,0.0])
        # finish = rospy.get_time()
        # print start
        # print finish
            
        
    def navigate(self, coordinates):
        print 'Navigating to position: ', coordinates
        trans = Point(*(coordinates[0], coordinates[1], 0.0))
        rot = Quaternion(*(quaternion_from_euler(0.0, 0.0, 0.0, axes='sxyz')))
        pose = Pose(*(trans, rot))
        self.goal.target_pose.pose = pose
        self.goal.target_pose.header.stamp = rospy.Time.now()
        self.move_base.send_goal(self.goal)
        finished_within_time = self.move_base.wait_for_result(rospy.Duration(100.0))        
        if not finished_within_time:
            self.move_base.cancel_goal()  
            state =  self.move_base.get_state()
        else:
            state = self.move_base.get_state()            
        
        if state == 3:
            return True
        else:
            return False
    
    def action_clean(self):
        start = rospy.Time.now()
        self.twist.angular.z = 0.3
        print 'Cleaning'
        while (rospy.Time.now() - start)< rospy.Duration(8.0):
            # pass
            self.cmd_pub.publish(self.twist)
            self.r.sleep()
        self.cmd_pub.publish(Twist())
    
    def initial_pose(self, coordinates):
        self.initial_pub = rospy.Publisher('initialpose', PoseWithCovarianceStamped, queue_size=10)
        self.pose_cov = PoseWithCovarianceStamped()
        self.pose_cov.header.frame_id = 'map'        
        trans = Point(*(coordinates[0], coordinates[1], 0.0))
        rot = Quaternion(*(quaternion_from_euler(0.0,0.0,-1.57, axes='sxyz')))
        covariance = [0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06853891945200942]
        pose = PoseWithCovariance(*(Pose(*(trans, rot)), covariance))
        self.pose_cov.header.seq = rospy.Time.now()
        self.pose_cov.pose = pose
        k = 0
        while not rospy.is_shutdown():
            self.initial_pub.publish(self.pose_cov)
            rospy.sleep(1.0)
            k +=1
            if k>2:
                break
    
    def show_path(self,cells =[]):
        # print self.pose[cells]
        poses = []
        for cell in cells:
            poses.append(self.pose[cell])
        
        Line().publish_marker(pose=poses)
    
    def random_clean(self):
        positions = []
        while not rospy.is_shutdown():
            cell = np.random.choice(range(375,623))
            print cell
            if self.navigate(self.pose[cell]):
                self.action_clean()
            else:
                pass
if __name__ == '__main__':    
    # Path().visualize_dirt('/home/mtb/Documents/data/dirt_extraction_2/data/high/test2017-02-05_estimation.csv')
    # cells = [144, 481, 57, 387, 305, 423, 308, 412, 501, 422, 544, 415, 431, 441, 473, 597, 559, 411, 102, 363, 174, 490, 264, 38, 486, 500, 496, 350, 540, 376, 449, 109, 434, 246, 443, 284, 124, 475, 123, 180, 479, 615, 619, 522, 154, 273, 405, 206, 176, 76, 317, 406, 505, 226, 383, 557, 58, 599, 303, 84, 618, 2, 362, 419, 36, 230, 524, 297, 97, 583, 295, 606, 86, 510, 518, 10, 286, 393, 384, 497, 67, 82, 4, 216, 430, 331, 205, 506, 520, 429, 502, 106, 160, 358, 403, 63, 314, 608, 103, 135, 374, 553, 598, 79, 538, 107, 70, 575, 602, 456, 596, 161, 69, 282, 188, 24, 147, 40, 460, 199, 270, 455, 442, 137, 227, 187, 169, 487, 576, 414, 503, 551, 484, 603, 1, 417, 71, 222, 397, 89, 377, 6, 357, 126, 52, 469, 153, 301, 541, 235, 18, 240, 138, 523, 13, 605, 119, 198, 53, 34, 436, 554, 268, 158, 607, 332, 193, 173, 253, 345, 369, 580, 375, 527, 555, 408, 195, 327, 396, 476, 133, 309, 438, 306, 55, 233, 207, 118, 64, 572, 272, 44, 604, 45, 410, 620, 326, 73, 342, 366, 611, 210, 400, 440, 349, 116, 329, 267, 111, 7, 59, 145, 344, 95, 371, 418, 485, 533, 313, 491, 214, 477, 556, 364, 370, 594, 617, 72, 258, 181, 565, 347, 289, 509, 399, 563, 203, 459, 381, 104, 175, 424, 269, 241, 609, 409, 360, 373, 395, 277, 162, 3, 127, 519, 25, 328, 16, 51, 132, 333, 94, 263, 337, 296, 9, 101, 31, 48, 351, 278, 352, 437, 413, 382, 568, 183, 219, 29, 562, 612, 237, 394, 77, 514, 43, 445, 66, 218, 293, 139, 592, 463, 265, 493, 466, 60, 480, 302, 450, 35, 546, 315, 228, 244, 584, 462, 391, 335, 513, 401, 14, 32, 316, 567, 386, 165, 614, 465, 454, 200, 359, 113, 19, 256, 616, 65, 452, 242, 590, 600, 260, 304, 504, 330, 507, 81, 420, 595, 495, 299, 39, 561, 150, 353, 151, 402, 114, 5, 321, 311, 191, 447, 300, 121, 288, 285, 446, 28, 131, 27, 336, 536, 548, 488, 593, 566, 254, 166, 75, 432, 201, 348, 457, 141, 492, 433, 172, 194, 248, 152, 340, 68, 398, 458, 539, 467, 99, 385, 11, 472, 404, 407, 62, 8, 494, 388, 427, 585, 515, 276, 37, 338, 365, 529, 223, 552, 310, 498, 128, 531, 613, 42, 380, 96, 319, 534, 341, 115, 134, 143, 92, 239, 444, 378, 392, 93, 15, 334, 171, 238, 197, 512, 543, 202, 17, 146, 56, 621, 87, 0, 435, 356, 167, 234, 582, 157, 483, 156, 564, 281, 250, 217, 261, 211, 426, 190, 184, 294, 186, 163, 189, 112, 368, 578, 318, 252, 117, 247, 579, 499, 535, 346, 298, 323, 215, 280, 125, 549, 209, 196, 185, 108, 168, 325, 461, 591, 236, 47, 588, 232, 21, 177, 221, 50, 33, 46, 287, 324, 571, 274, 526, 204, 255, 312, 20, 98, 225, 266, 88, 144]
    # cells = [144, 170, 274, 496, 438, 90, 607, 12, 547, 402, 48, 380, 461, 518, 425, 580, 76, 354, 382, 414, 601, 8, 267, 210, 213, 198, 91, 486, 539, 617, 307, 237, 436, 57, 610, 244, 368, 232, 151, 503, 47, 550, 271, 220, 320, 352, 618, 524, 45, 498, 21, 418, 184, 430, 544, 342, 335, 81, 215, 80, 459, 86, 118, 455, 351, 256, 312, 143, 142, 388, 56, 616, 462, 282, 400, 578, 75, 569, 452, 423, 202, 38, 39, 105, 521, 33, 440, 251, 581, 120, 266, 594, 94, 158, 145, 87, 600, 583, 223, 366, 457, 125, 197, 0, 283, 404, 221, 99, 71, 441, 113, 563, 127, 528, 279, 333, 500, 24, 344, 241, 512, 472, 327, 292, 612, 577, 212, 216, 343, 485, 475, 103, 261, 572, 129, 300, 311, 180, 317, 407, 83, 608, 248, 228, 14, 574, 11, 450, 519, 508, 106, 157, 305, 590, 92, 493, 506, 289, 84, 18, 36, 479, 467, 175, 481, 34, 357, 422, 478, 189, 253, 95, 473, 453, 464, 513, 365, 363, 52, 217, 37, 296, 182, 605, 490, 178, 477, 566, 41, 509, 287, 97, 98, 445, 207, 297, 54, 603, 599, 79, 242, 88, 132, 286, 389, 505, 4, 16, 176, 405, 229, 174, 155, 276, 255, 126, 265, 161, 322, 417, 413, 278, 69, 398, 225, 124, 119, 385, 514, 6, 412, 579, 379, 165, 593, 415, 470, 463, 138, 465, 150, 233, 573, 355, 32, 110, 367, 526, 211, 554, 309,
    #          347, 51, 520, 133, 208, 239, 65, 115, 112, 152, 499, 114, 531, 28, 390, 409, 619, 483, 60, 277, 303, 66, 49, 191, 535, 247, 339, 471, 224, 188, 454, 141, 22, 148, 123, 614, 270, 107, 206, 149, 375, 575, 540, 302, 219, 433, 341, 10, 516, 419, 245, 447, 551, 185, 318, 252, 373, 429, 374, 397, 268, 308, 214, 609, 193, 358, 263, 435, 394, 611, 330, 254, 348, 78, 474, 349, 596, 558, 439, 187, 58, 449, 383, 406, 437, 552, 315, 310, 480, 273, 293, 231, 284, 128, 564, 281, 108, 542, 458, 23, 549, 587, 416, 294, 329, 183, 553, 510, 264, 73, 19, 25, 431, 168, 491, 218, 560, 541, 326, 586, 70, 545, 591, 489, 396, 63, 61, 428, 190, 502, 272, 269, 523, 131, 1, 96, 360, 316, 146, 291, 410, 536, 288, 469, 44, 328, 15, 602, 162, 376, 192, 362, 194, 3, 346, 494, 262, 576, 167, 68, 582, 444, 42, 205, 324, 392, 147, 456, 259, 140, 117, 381, 384, 204, 139, 203, 421, 484, 156, 411, 340, 295, 548, 597, 35, 331, 391, 434, 135, 257, 177, 43, 468, 443, 565, 59, 179, 243, 451, 173, 336, 525, 559, 621, 622, 240, 555, 482, 246, 598, 446, 588, 604, 567, 154, 196, 313, 585, 17, 222, 238, 424, 460, 102, 532, 504, 387, 537, 546, 314, 538, 488, 615, 543, 5, 137, 111, 426, 122, 53, 235, 275, 595, 109, 301, 209, 77, 46, 442, 153]#low
    # cells = [144, 33, 162, 111, 237, 521, 2, 334, 215, 446, 15, 348, 597, 344, 495, 105, 412, 51, 543, 336, 400, 323, 73, 316, 179, 503, 321, 313, 18, 129, 451, 423, 611, 508, 289, 488, 395, 370, 75, 464, 201, 156, 461, 5, 208, 551, 435, 557, 99, 440, 252, 174, 274, 520, 195, 213, 581, 157, 61, 563, 113, 263, 409, 26, 197, 486, 539, 306, 389, 127, 482, 531, 86, 375, 457, 6, 108, 577, 176, 392, 340, 418, 493, 19, 84, 102, 184, 308, 191, 297, 571, 399, 591, 491, 459, 382, 394, 499, 513, 522, 401, 507, 359, 319, 64, 254, 134, 278, 453, 46, 114, 180, 620, 326, 82, 57, 505, 44, 55, 169, 242, 432, 354, 492, 154, 135, 227, 133, 37, 298, 542, 54, 47, 466, 598, 71, 518, 259, 498, 479, 265, 214, 91, 351, 50, 443, 305, 426, 303, 230, 475, 221, 497, 109, 586, 553, 339, 366, 246, 310, 90, 455, 318, 565, 593, 63, 88, 525, 608, 528, 534, 474, 36, 12, 268, 257, 13, 352, 456, 431, 172, 538, 346, 136, 575, 267, 408, 142, 141, 515, 567, 261, 266, 85, 315, 556, 564, 80, 121, 273, 1, 112, 53, 452, 424, 331, 377, 229, 437, 194, 473, 561, 462, 416, 420, 547, 387, 92, 94, 390, 124, 476, 200, 337, 241, 506, 428, 361, 232, 0, 52, 546, 403, 258, 203, 439, 130, 260, 380, 309, 45, 177, 445, 527, 11, 107, 465, 362, 35, 181, 178, 549, 558, 101, 535, 83, 619, 576, 617, 585, 175, 333, 299, 151, 170, 56, 369, 68, 240, 509, 327, 150, 280, 248, 226, 560, 296, 145, 183, 120, 40, 601, 566, 126, 592, 324, 81, 301, 376, 189, 304, 272, 524, 411, 10, 374, 48, 209, 244, 70, 510, 607, 122, 222, 364, 402, 87, 469, 433, 253, 353, 207, 155, 307, 275, 436, 34, 511, 216, 385, 512, 98, 211, 192, 595, 529, 381, 487, 472, 187, 530, 523, 552, 470, 228, 417, 28, 96, 347, 123, 317, 243, 444, 320, 618, 621, 283, 485, 219, 31, 269, 220, 449, 357, 438, 290, 104, 264, 407, 590, 458, 405, 25, 379, 504, 471, 610, 67, 132, 198, 532, 501, 3, 349, 602, 21, 425, 481, 519, 29, 62, 24, 182, 58, 330, 277, 540, 447, 613, 378, 238, 371, 578, 196, 384, 284, 550, 294, 343, 210, 489, 541, 116, 588, 478, 76, 89, 106, 65, 140, 300, 234, 39, 533, 72, 356, 363, 516, 186, 398, 587, 97, 95, 270, 239, 338, 66, 427, 622, 77, 224, 149, 484, 235, 202, 286, 579, 32, 582, 245, 559, 448, 285, 293, 434, 569, 345, 168, 609, 188, 589, 131, 406, 584, 419, 342, 249, 367, 480, 302, 137, 368, 78, 500, 537, 225, 159, 468, 4, 312, 223, 421, 388]

    # cells =[12, 22, 23, 24, 25, 35, 43, 44, 52, 59, 75, 89, 104, 118, 134, 149, 163, 184, 183, 182, 181, 180, 179, 178, 177, 176, 175, 156, 155, 154, 140, 126, 112, 98, 83, 69, 55, 71, 87, 103, 119, 133, 147]
    #Low
    # cells = [12, 23, 24, 25, 36, 44, 37, 43, 52, 59, 74, 88, 102, 118, 134, 149, 165, 186, 207, 222, 208, 188, 189, 210, 223, 209, 224, 238, 252, 266, 279, 265, 264, 277, 290, 276, 262, 275, 274, 273, 259, 245, 231, 217, 203, 182, 162, 161, 146, 160, 179, 178, 177, 157, 176, 156, 175, 154, 139, 123, 108, 93, 78, 79, 80, 81, 96, 97, 83, 68, 54, 55, 71, 56, 48, 40, 32, 38, 30, 21, 20]
    
    #high
    # cells =[375,399, 398, 422, 447, 469, 491, 508, 519, 530, 539, 553, 567, 582, 598, 599, 597, 580, 579, 564, 563, 549, 548, 547, 546, 536, 526, 515, 514, 503, 481, 480, 458, 436, 411, 412, 413, 437, 438, 459, 460, 461, 482, 483, 484, 462, 463, 464, 465, 443,375]#[312, 328, 326, 343, 359, 357, 373, 398, 397, 395, 419, 444, 466, 488, 486, 485, 484, 483, 503, 502, 501, 500, 477, 454, 452, 426, 424, 447, 469, 491, 508, 519, 530, 539, 553, 568, 551, 550, 549, 548, 547, 535, 534, 533, 541, 554]
#[299,312, 310, 309, 307, 324, 323, 322, 321, 337, 336, 335, 334, 352, 351, 350, 348, 346, 344, 361, 359, 358, 374, 398, 422, 447, 446, 467, 466, 488, 464, 486, 484, 483, 482, 504, 503, 514, 513, 512, 511, 492, 491, 508, 519, 529, 538, 547, 562, 561, 560, 559, 558, 572, 586, 584, 583, 581, 597, 579]
    Path().random_clean()
    # Path().show_path(cells=cells)
    # print cells.index(465)
    # print len(cells)
    # Path().clean_cells(cells=cells,starts=50)    

