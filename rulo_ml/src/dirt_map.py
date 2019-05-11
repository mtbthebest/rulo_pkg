#!/usr/bin/env python 

import os
os.environ['TF_CPP_MIN_LOG_LEVLEL'] = '2'
import tensorflow as tf
import numpy as np
from collections import deque, OrderedDict
from csvreader import csvread
from csvcreater import csvcreater
from csvwriter import csvwriter
learning_rate = 0.001
training_epochs = 2000
batch_size = 1000

n_inputs = 720
n_hidden1=400
n_hidden2=400
n_hidden3=500
n_outputs = 1296
model_path =os.path.abspath(os.path.join(os.path.dirname(__file__),"..")) + '/model' 
TRAIN_FILE_DIR ='/home/barry/hdd/data/train/'
SUM_DIR = str(model_path) + '/' + 'summary/'
TRAIN_MODEL_DIR = str(model_path) + '/' + 'train'
fieldnames = ['accuracy', 'iterations']
csvcreater(TRAIN_MODEL_DIR + '/' + 'evaluation.csv')
graph = tf.Graph()

with graph.as_default():

    with tf.name_scope('placeholder'):
        x = tf.placeholder(dtype=tf.float32, shape=[None,n_inputs],name ='laser')
        y = tf.placeholder(dtype=tf.float32, shape=[None,n_outputs], name='dirt_level')

    with tf.name_scope('layers'):
        hidden1 = tf.layers.dense(x, n_hidden1,activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), name='hidden1')
        hidden2= tf.layers.dense( hidden1,n_hidden2,activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), name='hidden2')
        hidden3= tf.layers.dense( hidden2,n_hidden3,activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), name='hidden3')
        out =  tf.layers.dense( hidden3,n_outputs,activation=tf.nn.relu, name='output')
    
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.square(tf.subtract(y, out)))
        tf.summary.scalar('loss', loss)

    with tf.name_scope('optimizer'):
        global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step)
    
    with tf.name_scope('accuracy'):
        correct_pred = tf.equal(y, out)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

   
    with tf.name_scope('initializer'):
       init = tf.global_variables_initializer()
       saver = tf.train.Saver()
       summary_op = tf.summary.merge_all()
input_ = np.load(TRAIN_FILE_DIR + 'input.npy')
output_ = np.load(TRAIN_FILE_DIR + 'output.npy')
with tf.Session(graph=graph) as sess:
   
    input_dict = OrderedDict()
    output_dict = OrderedDict()
    for k in range(800):
        start = k*batch_size
        stop = (k+1)*batch_size
        input_dict[str(k)] = deque()
        output_dict[str(k)] = deque()
        for i in range(start, stop):
            input_dict[str(k)].append(input_[i])
            output_dict[str(k)].append(output_[i])
    
    valid_input = []
    valid_output = []
    for j in range(800100,800400):
        valid_input.append(input_[i])
        valid_output.append(output_[i])
        
    summary_writer = tf.summary.FileWriter(logdir= SUM_DIR, graph = graph)
    sess.run([init])
    if os.path.isfile(TRAIN_MODEL_DIR + '/' + str('checkpoint')):
            print 'Restoring the Variables'
            saver.restore(sess,'/home/barry/hdd/rulo_ml/model/train/model.ckpt-1600000')
    iterations = 0
    for epoch in range(training_epochs):
        for j in range(800): 
            batch_x, batch_y = np.asarray(input_dict[str(k)]),np.asarray(output_dict[str(k)])
            train, cost, summary = sess.run(fetches =  [train_op, loss, summary_op], feed_dict={x:batch_x, y: batch_y})    
            
            iterations +=1
            
            step = tf.train.global_step(sess, global_step)
            summary_writer.add_summary(summary, step)       
                
            if(iterations%1000==0):            
                saver.save(sess, str(TRAIN_MODEL_DIR +'/'+'model.ckpt'), global_step)
                print 'Iterations: ' + str(iterations) + ' , Cost: ' + str(cost)
        
        y_predict, acc , summary= sess.run([out, accuracy,summary_op], feed_dict={x:np.asarray(valid_input), y: np.asarray(valid_output)})
        print acc
        summary_writer.add_summary(summary, step)
        csvwriter(TRAIN_MODEL_DIR + '/' + 'evaluation.csv', fieldnames, row={'accuracy': acc, 'iterations': iterations})
            






