#!/usr/bin/env	python
import os
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from collections import OrderedDict
from tensorflow.contrib.layers import variance_scaling_initializer

class CNN:
    
    def __init__(self):
        self.input  = np.load('/home/mtb/rulo_ws/src/rulo_pkg/rulo_ml/train_data/inputs.npy')
        self.output  = np.load('/home/mtb/rulo_ws/src/rulo_pkg/rulo_ml/train_data/outputs.npy')
        # pass
    def get_input(self):
        pose_file = '/home/mtb/rulo_ws/src/rulo_pkg/rulo_ml/train_data/positions.npy'
        self.pose = np.load(pose_file)
        air_folder = '/home/mtb/Documents/data/dirt_extraction_2/result/air_influence/dirt_variations/no_scaling/low/cells/'
        human_folder = '/home/mtb/Documents/data/dirt_extraction_2/result/human_influence/human_data/high/data/'
        result = OrderedDict()
        for files in sorted(os.listdir(air_folder)):
            result[files[:-4]] = OrderedDict()            
            data = pd.read_csv(air_folder + files)
            result[files[:-4]]['pose'] = self.pose[int(files[:-4])]
            result[files[:-4]]['dirt_level'] = data['dirt_level'].values[-8:]
            result[files[:-4]]['cleaning_cycle'] = data['duration'].values[-8:]
            result[files[:-4]]['weights'] = np.array([0.0]* 8)
           
        folders =[]
            
        for files in sorted(os.listdir(human_folder)):  
            if files[:-4] not in result:
                
                data = pd.read_csv(human_folder + files, index_col=[0])
                
                if data['dirt_level'].values.shape[0]>=8:
                    result[files[:-4]] = OrderedDict()
                    result[files[:-4]]['pose'] = self.pose[int(files[:-4])]
                    result[files[:-4]]['dirt_level'] = data['dirt_level'].values[-8:]
                    result[files[:-4]]['cleaning_cycle'] = data.index.values[-8:]
                    result[files[:-4]]['weights'] = 0.001* (data['human_frequency'].values[-8:] / data.index.values[-8:])
            else:
                pass
       
        iteration = 0
        for cells in result:
            for i in range(8):
                # iteration +=1
        # print iteration
                try:
                    self.input = np.concatenate((self.input,[np.array([list(result[cells]['pose']),
                                                                       result[cells]['weights'][i], 
                                                                       result[cells]['cleaning_cycle'][i]])]))  # ,result[cells]['dirt_level'][i]])
                    self.output = np.concatenate((self.output,np.array([result[cells]['dirt_level'][i]])))
                except:
                       self.input = np.array([[list(result[cells]['pose']),
                                            result[cells]['weights'][i],
                                            result[cells]['cleaning_cycle'][i]]])#,result[cells]['dirt_level'][i]])
                       self.output = np.array([result[cells]['dirt_level'][i]])

        # print self.input.shape
        # # print len(result.keys())
        # np.save('/home/mtb/rulo_ws/src/rulo_pkg/rulo_ml/train_data/inputs.npy', self.input)
        # np.save('/home/mtb/rulo_ws/src/rulo_pkg/rulo_ml/train_data/outputs.npy', self.output)

    def train(self):
        pose= tf.placeholder(dtype=tf.float32, shape=[None,2])
        weights = tf.placeholder(dtype=tf.float32, shape=[None,1])
        duration = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        inputs = tf.concat([pose, weights,duration], axis=1)
        outputs = tf.placeholder(tf.float32, shape=[None,65])

        net1 = tf.layers.dense(inputs, 500, activation=tf.nn.relu,
                               kernel_initializer=variance_scaling_initializer())
        
        net2 = tf.layers.dense(net1, 400, activation=tf.nn.relu,
                               kernel_initializer=variance_scaling_initializer())

        net3 = tf.layers.dense(net2, 300, activation=tf.nn.relu,
                               kernel_initializer=variance_scaling_initializer())
        
        net4 = tf.layers.dense(net3, 1, activation=tf.nn.relu,
                               kernel_initializer=variance_scaling_initializer())
        
        net5 = tf.reshape(net4,[1,65])
    
        loss = tf.reduce_mean(tf.pow(tf.subtract(outputs , net5), 2))
        
        optimize = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)
        tf.summary.scalar('loss ', loss)
        summary_op = tf.summary.merge_all()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            positions= []
            for elem in np.array(self.input[:, 0]):
                positions.append(elem)
            for i in range(4328):
                self.input[:, 1][i] = [self.input[:, 1][i]]
                self.input[:, 2][i] = [self.input[:, 2][i]]
                # self.output[i] = [self.output[i]]
            wei = []
            dur = []
            # out
            for elem in np.array(self.input[:, 1]):
                wei.append(elem)            
            for elem in np.array(self.input[:, 2]):
                dur.append(elem)
            summary_writer = tf.summary.FileWriter(
                logdir='/home/mtb/rulo_ws/src/rulo_pkg/rulo_ml/summary/')
            for episodes in range(1000000):
                for k in range(65):
            
                    opt,err,summary= sess.run([optimize, loss, summary_op], {pose: positions[k * 65: (k + 1) * 65], 
                                        weights: wei[k * 65: (k + 1) * 65], 
                                        duration: dur[k * 65: (k + 1) * 65],
                                            outputs:self.output[k * 65: (k + 1) * 65].reshape(1,65)})

                    print err
                               
                    summary_writer.add_summary(summary)

                saver.save(sess,'/home/mtb/rulo_ws/src/rulo_pkg/rulo_ml/cnn_var/model.ckpt')
                
                
                    
            
        
if __name__ == '__main__':
    CNN().train()
    # CNN().get_input()

    
