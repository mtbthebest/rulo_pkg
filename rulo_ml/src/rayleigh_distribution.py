#!/usr/bin/env	python
import os
import sys
import tensorflow as tf
import numpy as np
from rulo_utils.csvreader import csvread
from rulo_utils.csvwriter import csvwriter
from math import pi, sqrt
from rulo_utils.graph_plot import Plot
import matplotlib.pyplot as plt
filepath = '/home/mtb/12-25/human influence/human_and_dirt_high.csv'
class Distribution:
    def __init__(self):
        self.data = csvread(filepath)
        self.human_presence = self.data['human_presence']
        self.dirt_level = self.data['dirt_level']
        for i in range(len(self.human_presence)):
            try:
                self.array = np.concatenate((self.array, [{str(i): [float(self.human_presence[i]),
                                                                    int(self.dirt_level[i])]}]))
            except:
                self.array = np.array([{str(i): [float(self.human_presence[i]),
                                                 int(self.dirt_level[i])]}])
    
    def evaluate(self):
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
        plt.plot(mean_presence, mean_dirt, 'ro')
        plt.show()
        
    
            # sigma = mean_dirt * sqrt(2/pi)
            # csvwriter(filepath='/home/mtb/12-25/human influence/rayleigh_high_level_sensor_108samples.csv',
            #           headers=['mean_dirt', 'mean_presence','sigma'], 
            #           rows = [[mean_dirt],[mean_presence],[sigma]])
    
    def plot_sigma_presence(self):
        filename = '/home/mtb/12-25/human influence/rayleigh_high_level_sensor_108samples.csv'
        data = csvread(filename)
        x = data['mean_presence']
        y = data['sigma']
        w = [float(x[i]) for i in range(len(x))]
        z= [float(y[i]) for i in range(len(y))]
        for k in range(100):
            plt.plot(w[k * 108:(k + 1) * 108], z[k * 108:(k + 1) * 108], 'ro')
            plt.show()
        
    def find_sigma(self):
        for k in range(1000):
            print k            
            self.evaluate()

    def get_air_influence_parameters(self):
        duration = [0, 22, 5, 62, 51, 5, 20, 7, 62, 8, 16, 8, 18, 7, 18, 25, 7, 88, 7, 16, 24,
                290, 23, 48, 96, 99, 69, 27, 23, 119, 22, 24,
                50, 45, 27, 46, 49, 72, 23, 25, 27, 95, 31, 18]

        filename = '/home/mtb/12-25/air influence/air_influence_high.csv'
        data = csvread(filename)        
        time= [sum(duration[0:j]) for j in range(len(duration))]        

        y = tf.placeholder(dtype=tf.float32,shape=(None,))
        t = tf.placeholder(dtype=tf.float32, shape=(None,))
        y0 = tf.Variable(initial_value=0.01)
        p = tf.Variable(initial_value= 500.0)
        k = tf.Variable(initial_value=100000.0)
        y_predicted = tf.multiply(y0,tf.exp(tf.divide(t,k))) - p
        loss = y - y_predicted
        opt = tf.train.GradientDescentOptimizer(learning_rate =0.001)#.minimize(loss)
        grads_and_vars= opt.compute_gradients(loss,[y0,k,p])
        optimize = opt.apply_gradients(grads_and_vars)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for n in range(2000):
                for keys in data:
                    datas = [int(data[keys][i])
                             for i in range(len(duration))]
                    y_actual = [float(datas[i]) / float(max(datas))
                                for i in range(len(datas))]
                    # print y_actual
                    # for i in range(len(time)):
                    # error, y_act, y_pred = sess.run(
                    #     [loss, y, y_predicted], {t: time, y: y_actual})
                    # print error
                    # print y_act
                    # print y_pred
                    error , optimizer, var = sess.run([loss,optimize, y0], {t: time, y: y_actual})
                    print error
         
    def view_distribution(self):
        filename = '/home/mtb/12-25/human influence/rayleigh_low_level_sensor_108samples.csv'
        data = csvread(filename)
        x = data['mean_presence']
        y = data['mean_dirt']
        w = [float(x[i]) for i in range(len(x))]
        z = [float(y[i]) for i in range(len(y))]
        for k in range(100):
            plt.plot(w[k * 108:(k + 1) * 108], z[k * 108:(k + 1) * 108], 'ro')
            plt.show()

if __name__ == '__main__':
    
    # Distribution().find_sigma()
    # Distribution().plot_sigma_presence()
    # Distribution().get_air_influence_parameters()
    # Distribution().view_distribution()
    Distribution().evaluate()
