#!/usr/bin/env python
import os
import rospy
os.environ['TF_CPP_MIN_LOG_LEVLEL'] = '2'

import numpy as np
from collections import deque, OrderedDict
from rulo_utils.csvreader import csvread
from rulo_utils.csvconverter import csvconverter
from rulo_utils.numpyreader import numpyreader
from rnn_data import Process
from rulo_base.markers import VizualMark, TextMarker
eval_file_correct = '/home/mtb/rulo_ws/src/rulo_pkg/rulo_ml/model/evaluation/correct_labels.csv'
eval_file_predict = '/home/mtb/rulo_ws/src/rulo_pkg/rulo_ml/model/evaluation/predict_labels.csv'
precision_file = '/home/mtb/rulo_ws/src/rulo_pkg/rulo_ml/model/evaluation/precision.csv'
class Reader:
    def __init__(self):
        pass

    def get_data(self):
        self.correct = numpyreader(eval_file_correct).reshape((-1,1296,3))
        self.predict = numpyreader(eval_file_predict).reshape((-1,1296,3))
    
    def get_pose(self, indice):
        self.get_data()
        self.predict_pose_array = self.predict[indice]
        self.correct_pose_array = self.correct[indice]
        self.center_pose = Process().get_center()
        return self.correct_pose_array, self.predict_pose_array, self.center_pose
    
    def get_argmax(self, indice=500):
        self.get_pose(indice)
        self.correct_pose_arg = np.argmax(self.correct_pose_array, axis=1)
        self.predict_pose_arg = np.argmax(self.predict_pose_array, axis=1)
        return self.correct_pose_arg, self.predict_pose_arg
        
    def get_marker_pose(self,indice=500):
        self.get_argmax(indice)
        self.correct_pose_marker_arg = np.where(self.correct_pose_arg<2)[0]
        self.predict_pose_marker_arg = np.where(self.predict_pose_arg < 2)[0]
        self.accuracy = 0
        self.correct_pose = [self.center_pose[arg] 
                             for arg in self.correct_pose_marker_arg]
        self.predict_pose = [self.center_pose[arg]
                             for arg in self.predict_pose_marker_arg]
        for arg in self.correct_pose_marker_arg:
            if arg in self.predict_pose_marker_arg:
                self.accuracy +=1
        
        self.accuracy_text = float(self.accuracy) / float(self.correct_pose_marker_arg.shape[0])
        

    # def get_precision(self, indice =500):
    #     data = csvread(precision_file)
    #     return 
        
    def rviz(self,indice=500, labels ='correct'):
        self.get_marker_pose(indice)
        if labels =='correct':
            pose = self.correct_pose
            color = ['Red']
        if labels =='predict':
            pose = self.predict_pose  
            color = ['Green']      
        TextMarker().publish_marker(text_list=['accuracy: '+str(self.accuracy_text), 'iterations: ' +str(indice *1000)], pose=[[3.0,-2.0], [2.5,-2.0]])
        VizualMark().publish_marker(pose, 
                                    sizes=[[0.25,0.25,0.0]]*len(pose),
                                     color=color*len(pose))

if __name__ == '__main__':
    rospy.init_node('reader')
    Reader().rviz(indice=140, labels='predict')
    # a  = np.array([1,2,3,4,5,10,7,10])
    
    # print np.where(a<10)[0]
    # print np.argmax(a, axis=1)
