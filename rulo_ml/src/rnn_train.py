#!/usr/bin/env python  
 
import os 
os.environ['TF_CPP_MIN_LOG_LEVLEL'] = '2' 
import tensorflow as tf 
from tensorflow.contrib.rnn import BasicRNNCell, BasicLSTMCell, MultiRNNCell
import numpy as np 
from collections import deque, OrderedDict 
from rulo_utils.csvreader import csvread 
from rulo_utils.csvcreater import csvcreater 
from rulo_utils.csvwriter import csvwriter 
from rulo_utils.numpywriter import numpywriter
learning_rate = 0.001 
training_epochs = 10000
batch_size = 36 
total_batch = 36
n_inputs = 11
n_steps = 19
n_class = 2
n_lstm_neurons = 100
n_layers = 3

model_path =os.path.abspath(os.path.join(os.path.dirname(__file__),"..")) + '/model'  
TRAIN_FILE_DIR = '/home/mtb/Documents/data/features/11-24/'
SUM_DIR = model_path + '/summary' 
EVAL_DIR = model_path + '/evaluation/'
TRAIN_MODEL_DIR = model_path + '/train/' 

# fieldnames = ['accuracy', 'iterations'] 
# csvcreater(TRAIN_MODEL_DIR '/evaluation.csv') 
graph = tf.Graph() 
 
with graph.as_default(): 
    def lstm_cell():
        lstm_cell = BasicRNNCell(num_units = n_lstm_neurons) 
        return lstm_cell

    with tf.name_scope('inputs'):  
        X = tf.placeholder(dtype=tf.float32, shape=[None,n_steps,n_inputs]) 
        y = tf.placeholder(dtype=tf.float32, shape=[None, n_class])

    with tf.name_scope('rnn_dynamic'):        
        lstm_layer_cell = MultiRNNCell([lstm_cell() for _ in range(n_layers)])
        lstm_outputs, lstm_states = tf.nn.dynamic_rnn(lstm_layer_cell, X, dtype =tf.float32)
        lstm_out_tansp = tf.transpose(lstm_outputs, [1,0,2])
        lstm_last_batch_output = tf.gather(lstm_out_tansp, int(lstm_out_tansp.get_shape()[0]) - 1)
        predicted_output = tf.layers.dense(lstm_last_batch_output, n_class, activation=tf.nn.tanh)
    
    with tf.name_scope('loss'):
        error = tf.subtract(y,predicted_output)
        cost_function = tf.reduce_mean(tf.square(error))
        tf.summary.scalar('loss' ,cost_function)
    
    with tf.name_scope('optimizer'):
        global_step =tf.Variable(initial_value=0, trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(cost_function, global_step)

    with tf.name_scope('evaluation'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, axis=1), tf.argmax(predicted_output, axis=1)), tf.float32))
        actual_negative = tf.equal(tf.argmax(y, axis=1), 1)
        actual_positive = tf.equal(tf.argmax(y, axis=1), 0)
        predict_positive = tf.equal(tf.argmax(predicted_output, axis=1), 0)
        predict_negative = tf.equal(tf.argmax(predicted_output, axis=1), 1)

        true_negative = tf.reduce_sum(tf.cast(tf.logical_and(actual_negative, predict_negative), tf.int32))
        true_positive =tf.reduce_sum(tf.cast(tf.logical_and(actual_positive, predict_positive), tf.int32))
        false_negative=tf.reduce_sum(tf.cast(tf.logical_and(actual_positive, predict_negative), tf.int32))
        false_positive = tf.reduce_sum(tf.cast(tf.logical_and(actual_negative, predict_positive), tf.int32))
        numerator = true_positive
        precision_denominator = tf.add(numerator, false_positive)
        recall_denominator = tf.add(numerator, false_negative)
        precision_op = tf.divide(numerator, precision_denominator)
        recall_op = tf.divide(numerator,recall_denominator)        
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('precision', precision_op)
        tf.summary.scalar('recall', recall_op)


    with tf.name_scope('initializer'):
        init = tf.global_variables_initializer()
        summary_op = tf.summary.merge_all()
        saver = tf.train.Saver()

  
with tf.Session(graph=graph) as sess:      
    sess.run(init)
    train_features= np.load(TRAIN_FILE_DIR + 'train_features.npy') 
    train_labels = np.load(TRAIN_FILE_DIR  + 'train_labels.npy')
    test_features = np.load(TRAIN_FILE_DIR + 'test_features.npy')
    test_labels = np.load(TRAIN_FILE_DIR  + 'test_labels.npy')
    summary_writer = tf.summary.FileWriter(SUM_DIR, graph)    
    iterations = 0   
    for epoch in range(training_epochs):
        for k in range(batch_size):
            batch_start = k * 36 
            batch_stop = (k + 1) * 36
            batch_x, batch_y = train_features[batch_start:batch_stop], train_labels[batch_start:batch_stop]
            iterations +=1
            train, cost, summary= sess.run([train_op,cost_function, summary_op],  {X: batch_x, y: batch_y})        
            summary_writer.add_summary(summary,tf.train.global_step(sess, global_step))
            if (iterations%1000) == 0:
                acc,y_pred,precision, recall,summary = sess.run([accuracy,predicted_output, precision_op,recall_op, summary_op],  
                                               {X: test_features, y: test_labels})
                summary_writer.add_summary(summary, tf.train.global_step(sess, global_step))
                csvwriter(EVAL_DIR + 'precision.csv', ['precisions', 'recall', 'iterations'], [[precision], [recall],[iterations]])
                numpywriter(EVAL_DIR + 'correct_labels.csv' , test_labels)
                numpywriter(EVAL_DIR + 'predict_labels.csv' , y_pred) 
                print precision      , recall         
        saver.save(sess, TRAIN_MODEL_DIR + 'model.ckpt', global_step )
        print 'Model Saved.'
        print 'Epoch: '  + str(epoch)

   
