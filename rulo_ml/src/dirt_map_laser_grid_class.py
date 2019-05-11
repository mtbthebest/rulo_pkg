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
training_epochs = 1000
batch_size = 1000
TRAIN_DATA_NUM = 1500000
EVAL_DATA_NUM=5000
n_inputs = 720
n_hidden1=400
n_hidden2=400
n_hidden3=500
n_outputs = 1296
model_path =os.path.abspath(os.path.join(os.path.dirname(__file__),"..")) + '/model_laser_grid_class' 
TRAIN_FILE_DIR ='/home/barry/hdd/data/train/'
SUM_DIR = str(model_path) + '/' + 'summary/'
TRAIN_MODEL_DIR = str(model_path) + '/' + 'train'
fieldnames = ['labels', 'predicted']
csvcreater(TRAIN_MODEL_DIR + '/' + 'accuracy_prediction.csv', fieldnames)
graph = tf.Graph()
data_train = np.load(TRAIN_FILE_DIR + 'data_IO.npy')
np.random.shuffle(data_train) 
def create_batch(input_array, output_array, j):
    input_list , output_list = [],[]
    for in_ in input_array:
        if in_.shape ==(720,):
            input_list.append(np.array(in_))
        else:
            break
      
    for i  in range(len(input_list)):
        output_list.append(np.array(output_array[i]))
    return np.array(input_list), np.array(output_list)

def create_test_values():
    test_val = False
    while not test_val:
        array_choice = np.arange(TRAIN_DATA_NUM, TRAIN_DATA_NUM+EVAL_DATA_NUM, dtype=np.int32)
        eval_choice = np.random.choice(array_choice, size=1)[0]
        input_ = np.array(data_train[eval_choice][0], ndmin=2)
        output_ = np.array(data_train[eval_choice][1], ndmin=2)
        if input_.shape ==(1,n_inputs) and output_.shape ==(1,n_outputs):
            test_val = True
        else:
            test_val = False        
        return input_, output_

        
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




with tf.Session(graph=graph) as sess:
       
    summary_writer = tf.summary.FileWriter(logdir= SUM_DIR, graph = graph)
    sess.run([init])
    # if os.path.isfile(TRAIN_MODEL_DIR + '/' + str('checkpoint')):
    #         print 'Restoring the Variables'
    #         saver.restore(sess,'/home/barry/hdd/rulo_ml/model/train/model.ckpt-1600000')
    iterations = 0
    
    for epoch in range(training_epochs):
        print 'Epoch: ' + str(epoch) 
        for j in range(int(TRAIN_DATA_NUM/batch_size)): 
            batch_x, batch_y = create_batch(data_train[int(j*batch_size):int((j+1)*batch_size), 0], data_train[int(j*batch_size):int((j+1)*batch_size), 1], j)      
            if batch_x.shape[0] ==batch_size:
                train, cost, summary = sess.run(fetches =  [train_op, loss, summary_op], feed_dict={x:batch_x, y: batch_y})                
                iterations +=1                          
                if(iterations%500==0):                        
                    x_input, y_labels = create_test_values()                  
                    y_predict, acc , summary= sess.run([out, accuracy,summary_op], feed_dict={x:x_input, y: y_labels})        

                    csvwriter(TRAIN_MODEL_DIR + '/' + 'accuracy_prediction.csv', fieldnames, dict(zip(fieldnames, [list(y_labels[0]),list( y_predict[0])])))
                    saver.save(sess, str(TRAIN_MODEL_DIR +'/'+'model.ckpt'), global_step)
                    print 'Iterations: ' + str(iterations) + ' , Cost: ' + str(cost) + '  ,  Accuracy: ' + str(acc)
                step = tf.train.global_step(sess, global_step)
                summary_writer.add_summary(summary, step)       
        saver.save(sess, str(TRAIN_MODEL_DIR +'/'+'model.ckpt'), global_step)