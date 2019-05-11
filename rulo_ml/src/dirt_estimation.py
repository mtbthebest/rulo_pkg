#!/usr/bin/env	python
import os
import sys
import tensorflow as tf
import numpy as np
from rulo_utils.csvreader import csvread
from rulo_utils.csvwriter import csvwriter
from math import pi, sqrt, fabs
from rulo_utils.graph_plot import Plot
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame
from collections import OrderedDict

class AirInfluence:
    def __init__(self, sensor='high', estimation='linear'):
        # super(AirInfluence, self).__init__()
        self.sensor_type = sensor
        self.estimation_type = estimation
    
    def estimation(self):
        if self.sensor_type == 'high':
            filename = '/media/mtb/Data Disk/data/extraction/air influence/01-05/dirt_high_lev_increase.csv'
            data = pd.read_csv(filename, index_col=[0])
            # cells = ['839','841', '866','869']
            cells = ['839', '841', '869','842']

        elif self.sensor_type == 'low':
            filename = '/media/mtb/Data Disk/data/extraction/air influence/dirt_low_lev_increase.csv'
            data = pd.read_csv(filename, index_col=[0])
        
        if self.estimation_type =='linear':
            print 'Estimation air influence: ', self.estimation_type
            # x = tf.placeholder(tf.float32)
            # y = tf.pow(x,2)
            # gradients = tf.gradients(xs=x, ys=y)
            t = tf.placeholder(tf.float32)
            y_est = tf.placeholder(tf.float32)
            w = tf.Variable(initial_value=51.3301 )
            b = tf.Variable(initial_value=8818.89)
            y_pred = w*t+b
            loss = y_est - y_pred
            cost = tf.reduce_mean(tf.pow(loss,2))
            opt = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
            # gradients = tf.gradients(xs=[w], ys=cost)
            grads_and_vars = opt.compute_gradients(cost, [w,b])
            optimize = opt.apply_gradients(grads_and_vars)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                # res,grad = sess.run([gradients,y],{x: 7.0})
                # print res, grad
                # grad, error = sess.run([gradients, loss], {t: 1, y_est: 5.0})

                # grad , error= sess.run([gradients,loss], {t:1, y_est:5.0})
                # print grad, error
                # time_t = np.array([])
                # for times in data.index.values:
                #     time_t = np.concatenate((time_t, [3600.0*float(times)]), axis=0)
                time_t = data.index.values[-13:]
                # print time_t
                dirt_data= np.mean(data.values[-14:], axis=1)
                dirt_estimation_list = []#[np.mean(data.values[-14])]
                # print dirt_data
                
                for i in range(dirt_data.shape[0]-1):
                    dirt_estimation_list.append(dirt_data[i+1]-dirt_data[0])
                # print dirt_estimation_list
                duration=[0]
                #dirt_estimation = np.mean(dirt_data, axis=1)
                for i in range(time_t.shape[0]-1):
                    duration.append(time_t[i+1] - time_t[-13])

                print duration
                # print dirt_estimation_list
                # Plot().scatter_plot(duration, dirt_estimation_list,axis_max=[600,40000])
                m, p = 51.3301, 8818.9
                y = [m * duration[i] +p for i in range(len(duration))]
                print y
                # plt.plot(duration, dirt_estimation_list, 'ro',
                #          duration, y, 'g-', label='m, p = 51.3301, 8818.9')
                # plt.legend()
                # plt.show()
                
                # weight_list = []
                # error_list = []
                
                # for m in range(200000):
                #     # for i in range(1,len(dirt_estimation)):                   
                #         # error ,f= sess.run([loss,cost], {t: duration, y_est: dirt_estimation_list})
                #         grad_op, error, var1, var2= sess.run([optimize, loss,w,b], {t: duration, y_est: dirt_estimation_list})
                #         print error, var1, var2
                        # if not weight_list:
                        #     weight_list.append([var1, var2])
                        #     error_list.append(fabs(error))
                        # else:
                        #     if fabs(error) < error_list[0]:
                        #         weight_list[0] = [var1, var2]
                        #         error_list = [fabs(error)]
                        # print error, f

class HumanPresence:
    def __init__(self, sensor='high'):
        self.sensor_type = sensor
    
    def get_parameters(self):
        cleaning_cycle = [27, 46, 49, 72, 23, 25, 27, 95, 31, 18]
        for i in range(len(cleaning_cycle)):
            cleaning_cycle[i] = float(cleaning_cycle[i]) * 3600000.0
        filename = ['12_5.csv','12_7.csv','12_9.csv','12_12.csv','12_13.csv','12_14.csv',
                    '12_15.csv','12_19.csv','12_20.csv','12_21.csv']   
        param_dict = dict(zip(filename, cleaning_cycle))
        print param_dict
    def estimation(self, sample=False, find_distrib = False, distrib_filename=''):
        if self.sensor_type =='high':
            filename = '/media/mtb/Data Disk/data/extraction/human dirt/human_and_dirt_high.csv'
        elif self.sensor_type =='high':
            filename = '/media/mtb/Data Disk/data/extraction/human dirt/human_and_dirt_low.csv'
        
        data = csvread(filename)
        x_val = []
        y_val = []
        if sample:
            for i in range(len(data['human_presence'])):
                x_val.append(float(data['human_presence'][i]))
                y_val.append(float(data['dirt_level'][i]))
        # Plot().scatter_plot(x_val, y_val, axis_max=[10,40000])
        
            self.sample(x_val, y_val)
        if find_distrib:
            data = csvread(distrib_filename)
            for i in range(len(data['dirt'])):
                x_val.append(float(data['human'][i]))
                y_val.append(float(data['dirt'][i]))
            Plot().scatter_plot(x_val, y_val,'ro')
            
    def sample(self, x_val, y_val):
        self.x = x_val
        self.y = y_val
        for i in range(len(self.x)):
            try:
                self.array = np.concatenate((self.array, [{str(i): [float(self.x[i]),
                                                                    int(self.y[i])]}]), axis=0)
            except:
                self.array = np.array([{str(i): [float(self.x[i]),
                                                 int(self.y[i])]}])
       
        for sampling in range(100):
            self.sample = np.random.choice(self.array, (9, 1296))
            mean_dirt = []
            mean_presence = []
            for i in range(self.sample.shape[0]):
                human = []
                dirt = []
                for j in range(self.sample[i].shape[0]):
                    keys = self.sample[i][j].keys()[0]
                    human.append(self.sample[i][j][keys][0])
                    dirt.append(self.sample[i][j][keys][1])
                mean_dirt.append(np.mean(dirt))
                mean_presence.append(np.mean(human))
            try:
                Plot().scatter_plot(mean_presence, mean_dirt,marker='go')
            except KeyboardInterrupt:
                sys.exit(0)
            # csvwriter('/media/mtb/Data Disk/data/extraction/human dirt/hum_dirt_high.csv',
            #         headers =['dirt', 'human'], rows = [mean_dirt, mean_presence])

if __name__ == '__main__':
    # AirInfluence(sensor='high', estimation='linear').estimation()

    HumanPresence().get_parameters()
