#!/usr/bin/env	python
import os
import sys
import tensorflow as tf
from tensorflow.contrib.distributions import Gamma
from scipy.stats import gamma, expon
import numpy as np
from rulo_utils.csvreader import csvread
from rulo_utils.csvwriter import csvwriter
from rulo_utils.graph_plot import Plot
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame
from collections import OrderedDict
import seaborn as sns
dirt_sensor_file = '/home/mtb/Documents/data/dirt_extraction/'
reject = [447, 962, 967, 1178, 1200, 1201]
cells_file = '/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/data/corners2.csv'

air_cells = [838, 839, 840, 841, 842,
             865, 866, 867, 868, 869]
cells_num = 617
class DirtEstimation:
    def __init__(self,sensor='high'):
        self.sensor = sensor
    
    def estimate_air_inf(self, train = False):
        filepath = dirt_sensor_file + 'dirt_variation/' + self.sensor + '/cells/'
        self.air_duration = []
        self.air_dirt_level = []
        for files in os.listdir(filepath):
            # print files
            data = csvread(filepath + files)
            duration =data['duration']
            dirt_level = data['dirt_level']
            for i in range(len(duration)):
                self.air_duration.append(float(duration[i]))
                self.air_dirt_level.append(int(dirt_level[i]))
        
        # max_dirt = max(self.air_dirt_level)
        # self.func_distrib = []
        # for i in range(len(self.air_dirt_level)):
        #     val = float(self.air_dirt_level[i])/float(max_dirt)
        #     self.func_distrib.append(val)
        
        # print self.air_dirt_level
        
        mean = np.mean(self.air_duration)
        var = np.var(self.air_duration)
        self.alpha = mean ** 2 / var
        self.beta = mean / var
        print self.alpha, self.beta
        self.y = gamma.pdf(self.air_duration, a=self.alpha,scale = 1/self.beta)
    
        # print len(y)
        # print len(self.air_durastion)
        # dirt = [y[i] *10000.0/ max(y) for i in range(y.shape[0])]
        # print max(y)
        # print alpha, beta
        # print max(y), max(self.air_dirt_level)
        # Plot().scatter_plot(x_value= self.air_duration, y_value = dirt,show=True,)
        # print self.air_duration
        # plt.plot(self.air_duration, self.air_dirt_level, 'ro',
        #             self.air_duration,dirt, 'go')
        # plt.plot(list(self.air_duration), list(dirt),linestyle='--')
        # plt.show()
        if train:
            self.train(mode ='air',pdf=self.y)
        # self.sample()
      
    def sample(self,x_val,y_val,sample_size=None):
        # print sample_size
        for i in range(sample_size[0]*sample_size[1]):
            try:
                self.array = np.concatenate((self.array, [{str(i): [float(x_val[i]),
                                                                    float(y_val[i])]}]), axis=0)
            except:
                self.array = np.array([{str(i): [float(x_val[i]),
                                                 float(y_val[i])]}])     
        # for t in range(10):
        for sampling in range(10):
            self.sample = np.random.choice(self.array, (sample_size[0], sample_size[1]))
            # print self.sample
            mean_x =[]
            mean_y = []
            for i in range(self.sample.shape[0]):
                x = []
                y = []
                # mean_x = []
                # mean_y = []
                for j in range(self.sample[i].shape[0]):
                    keys = self.sample[i][j].keys()[0]
                    x.append(self.sample[i][j][keys][0])
                    y.append(self.sample[i][j][keys][1])
                mean_x.append(np.mean(x))
                mean_y.append(np.mean(y))
            try:
                print len(mean_x)
                Plot().scatter_plot(mean_x, mean_y, marker='bo',show=True)
                # Plot().scatter_plot(x, y, marker='bo', show=True)
            
                # print i                
            except KeyboardInterrupt:
                sys.exit(0)
    
    def train(self, pdf,mode='air' ):
            # x = tf.placeholder(tf.float32)
            dirt_est = tf.placeholder(tf.float32)
            est_y = tf.placeholder(tf.float32)        
            num_factor = tf.Variable(11411.0, trainable=True)
            den_factor = tf.Variable(2.0e-06, trainable=True)
            dirt_pred =  tf.divide(tf.multiply(est_y, num_factor), den_factor)
            loss = dirt_est - dirt_pred
            cost = tf.reduce_mean(tf.pow(loss, 2))
            opt = tf.train.GradientDescentOptimizer(learning_rate=0.0000000000000000001).minimize(cost)
            # grads_and_vars = opt.compute_gradients(cost, [num_factor])
            
            # optimize = opt.apply_gradients(grads_and_vars)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                # print self.air_dirt_level
                for t in range(10000):    
                    # print max(pdf)         
                    # a = sess.run([dirt_pred], {
                    #              est_y: pdf, dirt_est: self.air_dirt_level})
                    # print np.max(a)
                    grad, error,var1,var2= sess.run([opt, cost, num_factor,den_factor], {est_y: pdf, dirt_est: self.air_dirt_level})
                    print np.sqrt(error),var1, var2
    
    def check_estimation(self):
        self.estimate_air_inf()
        y = []
       
        b = gamma.pdf(self.air_duration, a=self.alpha, scale=1 / self.beta)
        # print b
        # m, p = 6000.0, 4.03922e-06
        m, p = 11411.0, 6.79954e-06
        for t in range(len(self.air_duration)):
            
            val = (m* b[t]) / p
            # print val
            y.append(val)
        # Plot().scatter_plot(self.air_duration, y, show=True)
        plt.plot(self.air_duration, self.air_dirt_level,'go',label='real data')#,
        plt.plot(self.air_duration, y,  'bo', label='estimation')
        plt.legend()
        plt.show()             
    
    def read_sample(self):
        self.sample = np.load('/home/mtb/sample.npy')
        for i in range(self.sample.shape[0]):
            if i in [1,4,9]:
                human = []
                dirt = []
                for j in range(self.sample[i].shape[0]):
                    keys = self.sample[i][j].keys()[0]
                    human.append(self.sample[i][j][keys][0])
                    dirt.append(self.sample[i][j][keys][1])
                # mean_dirt.append(np.mean(dirt))
                # mean_presence.append(np.mean(human))
                print human
                print dirt
                try:
                    Plot().scatter_plot(human, dirt, marker='go', show=True)
                    

                except KeyboardInterrupt:
                    sys.exit(0)

    def estimate_hum_inf(self):
        param_data = csvread(dirt_sensor_file + 'linear_param.csv')
        num_factor, div_factor, alpha, beta = float(param_data['num_factor'][0]), float(param_data['div_factor'][0]),\
                                              float(param_data['alpha'][0]),float(param_data['beta'][0])
        # print num_factor, div_factor, alpha, beta
        data = pd.read_csv(dirt_sensor_file + 'human_influence/estimation/' +
                       self.sensor + '/human_cell_weights.csv', index_col=[0])
        process =  data[data.values[:,1]>0.0]
       
        # print data.index.values
        human_dirt = []
        timestep = []
        total_dirt_list = []
        for i in range(process.index.values.shape[0]):
            air_dirt = self.get_air_dirt(process.index.values[i], alpha, beta) * num_factor / div_factor
            weights = process.values[i][1]
            total_dirt = process.values[i][0]
            total_dirt_list.append(total_dirt)
            dirt_est = (total_dirt - (1.0-weights)*air_dirt) / weights
            human_dirt.append(dirt_est)
            timestep.append(weights)# * process.index.values[i])
        # plt.plot(timestep, total_dirt_list, 'go', label='real data', markersize = 2)  
        # plt.title('Total amount of dirt sucked by the '+ self.sensor+ ' level sensor in function of weights')
        # plt.show()

        # plt.plot(timestep, human_dirt, 'go', label='real data', markersize = 2)  
        # plt.title('Human dirt estimation with the '+ self.sensor+ ' level sensor in function of weights')
        # plt.show()
        #  plt.plot(timestep, expt_est, 'bo', label='prediction', markersize =2)
        # print sorted(timestep)
        # Plot().scatter_plot(timestep,total_dirt_list, title='Dirt due to human presence',
        #                             labels =['time_presence', 'Dirt_level'],show=True)

        mean_lambda = 1.0/np.mean(timestep)
        print mean_lambda
        
        expt_est = []

        for t in range(len(timestep)):
            # 4.93 #* 3.517
            val = mean_lambda * np.exp(-timestep[t] * mean_lambda) * 3.517
            expt_est.append(val)
        # print expt_est
        # # plt.plot(timestep, total_dirt_list,'ro', timestep, expt_est, 'go')
        plt.plot(timestep, total_dirt_list, 'ro', label='train data', markersize = 4)  # ,
        plt.plot(timestep, expt_est, 'bo', label='prediction', markersize =4)
        plt.legend()
        plt.show()  
    
        
    
        # est = tf.placeholder(tf.float32)
        # dat = tf.placeholder(tf.float32)
        # epsilon = tf.Variable(3.51)
        # pred = epsilon  * dat
        # loss = est - pred
        # cost = tf.reduce_mean(tf.pow(loss, 2))
        # opt = tf.train.GradientDescentOptimizer(learning_rate=0.000001).minimize(cost)


        # with tf.Session() as sess:
        #     sess.run(tf.global_variables_initializer())
        #     for m in range(10000):
        #         opt_op, error, var= sess.run([opt, cost,epsilon], {dat:expt_est, est: total_dirt_list })
        #         print np.sqrt(error), var



        # Plot().scatter_plot(timestep, expt_est, title='Dirt due to human presence',
        #                                              labels =['time_presence', 'Dirt_level'],show=True)

        # self.sample(timestep, total_dirt_list, sample_size=(117,40))

        
        # Plot().scatter_plot(list(process.values[:, 1]), list(process.values[:, 0]), title='Dirt due to human presence',
        #                     labels =['weights_presence', 'Dirt_level'], axis_max=[0.4,20000],show=True)

        # self.sample(list(process.values[:, 1]), list(process.values[:, 0]), sample_size=(234,20))
        #     air_dirt.append(val2)
        # # print data['weights'].values
        # Plot().scatter_plot(list(data['weights'].values), air_dirt, show=True)


    
    def get_air_dirt(self,x,alpha, beta):
        return gamma.pdf(x, a=alpha, scale=1 / beta)


    def estimate_human_dirt(self):
        param_data = csvread(dirt_sensor_file + 'linear_param.csv')
        num_factor, div_factor, alpha, beta = float(param_data['num_factor'][0]), float(param_data['div_factor'][0]),\
            float(param_data['alpha'][0]), float(param_data['beta'][0])
        # print num_factor, div_factor, alpha, beta
        data = pd.read_csv(dirt_sensor_file + 'human_influence/estimation/' +
                           self.sensor + '/human_cell_weights.csv', index_col=[0])
        process = data[data.values[:, 1] > 0.0]

        # print data.index.values
        human_dirt = []
        timestep = []
        total_dirt_list = []
        for i in range(process.index.values.shape[0]):
            air_dirt = self.get_air_dirt(
                process.index.values[i], alpha, beta) * num_factor / div_factor
            weights = process.values[i][1]
            total_dirt = process.values[i][0]
            total_dirt_list.append(total_dirt)
            dirt_est = (total_dirt -  air_dirt)
            human_dirt.append(dirt_est)
            timestep.append(weights)
        plt.plot(timestep, human_dirt, 'ro', label='real data', markersize=2)
        plt.show()
        # plt.title('Human dirt estimation with the '+ self.sensor+ ' level sensor in function of weights')
    
    def pwm_data(self):
        data = csvread(dirt_sensor_file +'spin_low.csv')
        dirt_level = data['dirt_high_level']
        wall_time = data['wall_time']
        for i in range(len(dirt_level)):
            dirt_level[i]=int(dirt_level[i])
            wall_time[i] = float(wall_time[i])
        a = []
        b=[]
        times = []
        times.append(0.0)
        for i in range(len(dirt_level) - 1):
            times.append(wall_time[i + 1] - wall_time[i])
        
        for i in range(1,len(dirt_level)+1):
            a.append(sum(dirt_level[:i]))
            b.append(sum(times[:i]))

       
        
        
        Plot().scatter_plot(x_value= b, y_value = a , marker='ro', show=True)
        

if __name__ == '__main__':
        # DirtEstimation(sensor='low').estimate_air_inf(train=True)

    # DirtEstimation(sensor='high').estimate_hum_inf()
    # DirtEstimation(sensor='low').check_estimation()
    # DirtEstimation(sensor='high').read_sample()
     DirtEstimation(sensor='high').pwm_data()
