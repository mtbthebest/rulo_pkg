#!/usr/bin/env	python
import os
import sys
import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer
import numpy as np
import pandas as pd

from rulo_utils.csvwriter import csvwriter
from pandas import Series, DataFrame
from collections import OrderedDict
from Wander2 import Wander
from ReplayBuffer2 import Memory 



RANDOM_SEED = 1234
BUFFER_SIZE = 500
MINIBATCH_SIZE = 50

EPISODES = 100000
STEPS = 50
ACTOR_LEARNING_RATE = 0.001
CRITIC_LEARNING_RATE = 0.1
EXPLORE = 10000.0
GAMMA_FACTOR = 0.99

TAU = 0.01

NUM_INPUTS = 2

ACTOR_L1 = 400
ACTOR_L2 = 300



class DDPG(object):
    
    def __init__(self):
        pass  
    
    def softmax(self,x):
        den = 0.0
        for s in x[0]:
            den +=np.exp(s)
        for i in range(x[0].shape[0]):
            x[0][i] = np.exp(x[0][i])/den
        
        return x

    def execute(self, sess, actor, critic,train=True):
        sess.run(tf.global_variables_initializer())    
        saver = tf.train.Saver()   
        if os.path.isfile('/home/mtb/rulo_ws/src/rulo_pkg/rulo_ml/model/checkpoint'):
            saver.restore(sess, '/home/mtb/rulo_ws/src/rulo_pkg/rulo_ml/model/var.ckpt')
            print 'Restored'
                
        env = Wander()
        iterations = 0    
        if train: 
            actor.update_target_network()  
            critic.update_target_network()
            replaybuffer=Memory(BUFFER_SIZE,RANDOM_SEED)            
            iterations = 0    
            epsilon = 0.8
            for i in range(EPISODES):                
                s_t_d, s_t_p= env.reset() 
                self.last_position = 0
                total_reward = 0.0   
                explore = 0
                for j in range(STEPS):              
                    # epsilon =1- np.exp(-float(j) / (float(i) +1))
                    epsilon -=0.3/100000.0
                    iterations +=1                    
                    if np.random.random() > epsilon:
                        a_type = "Exploit"                        
                        # exploit +=1
                        a = actor.predict(s_t_d.reshape(1,50), s_t_p.reshape(1,50))                                                           
                    else:
                        a_type = "Explore"
                        explore +=1
                        act = np.random.randint(0, 9, 1)[0]
                        a = np.zeros([1, 9], dtype=float)
                        a[0][act] = 1.0                                          
                    action = np.argmax(a)   
                    s_t1_d, s_t1_p, r, new_position,terminal = env.step(self.last_position,s_t_d, s_t_p,action)                     
                    total_reward +=r   
                    replaybuffer.add(s_t_d, s_t_p ,a[0], r, terminal, s_t1_d, s_t1_p)   
                    if replaybuffer.size() >= BUFFER_SIZE:                        
                        s_batch_d, s_batch_p,a_batch, r_batch, t_batch, s2_batch_d, s2_batch_p = replaybuffer.sample_batch(MINIBATCH_SIZE)
                        target_q = critic.predict_target(np.array(s2_batch_d).reshape([-1, 50]),np.array(s2_batch_p).reshape([-1, 50]),np.array(a_batch).reshape([-1, 9]))
                        y_i = []
                        for k in xrange(BUFFER_SIZE):
                            if t_batch[k]:
                                y_i.append(r_batch[k])
                            else:
                                y_i.append(r_batch[k] + GAMMA_FACTOR * target_q[k])                        
                        predicted_q_value =critic.train(np.array(s_batch_d).reshape([BUFFER_SIZE, 50]),
                                                        np.array(s_batch_p).reshape([BUFFER_SIZE, 50]) ,
                                                        np.array(a_batch).reshape([BUFFER_SIZE,9]), 
                                                        np.reshape(y_i,(BUFFER_SIZE,1)))
                        loss = predicted_q_value[2]
                        a_outs = actor.predict(np.array(s_batch_d).reshape([BUFFER_SIZE, 50]),
                                            np.array(s_batch_p).reshape([BUFFER_SIZE, 50]))
                        grads = critic.action_gradients(np.array(s_batch_d).reshape([BUFFER_SIZE, 50]),
                                                        np.array(s_batch_p).reshape([BUFFER_SIZE, 50]) ,
                                                        a_outs)
                        actor.train(np.array(s_batch_d).reshape([BUFFER_SIZE, 50]),
                                    np.array(s_batch_p).reshape([BUFFER_SIZE, 50]),
                                    grads[0])
                        actor.update_target_network()
                        critic.update_target_network()                          
                
                    self.last_position = new_position
                    s_t_d , s_t_p= s_t1_d , s_t1_p

                    if terminal:
                        break   
                
                saver.save(sess, '/home/mtb/rulo_ws/src/rulo_pkg/rulo_ml/model/var.ckpt')   
                try:
                    print 'Episode %d , Reward: %.6f , Loss: %.7f  , Explore: %d'%(i,total_reward, loss,explore)
                except:
                    pass





class ActorNetwork(object):

    def __init__(self, sess, state_dim, action_dim,learning_rate, tau):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau

        self.actor_inputs_dirt, self.actor_inputs_presence ,self.actor_weights ,self.actor_out= self.create_actor_network('actor_network')

        self.target_actor_inputs_dirt,self.target_actor_inputs_presence, self.target_actor_weights ,self.target_actor_out = self.create_actor_network('actor_target')

        self.update_target_network_params = \
            [self.target_actor_weights[i].assign(tf.multiply(self.actor_weights[i], self.tau) +
                                                  tf.multiply(self.target_actor_weights[i], 1. - self.tau))
                                            for i in range(len(self.target_actor_weights))]

        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])
    
        self.actor_gradients = tf.gradients( self.actor_out,  self.actor_weights, -self.action_gradient)

        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(self.actor_gradients, self.actor_weights))


    def create_actor_network(self,scope_name):
        with tf.variable_scope(scope_name):
            I = tf.placeholder(tf.float32, shape=[None, self.s_dim ])
            P = tf.placeholder(tf.float32, shape=[None, self.s_dim ])
            H1 = tf.layers.dense(I, 400, activation=tf.nn.relu)
            H2 = tf.layers.dense(P, 400, activation=tf.nn.relu)
            H3 = tf.concat((H1,H2),axis=1)
            H4 = tf.layers.dense(H3,300,activation=tf.nn.relu)
            O = tf.layers.dense(H4,self.a_dim,activation=tf.nn.sigmoid)
        W = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)    
        return I , P, W , O

    def train(self, inputs, presence,a_gradient):
        self.sess.run(self.optimize, feed_dict={self.actor_inputs_dirt: inputs, self.actor_inputs_presence: presence, self.action_gradient: a_gradient})

    def predict(self, inputs,presence):
        return self.sess.run(self.actor_out, feed_dict={
            self.actor_inputs_dirt: inputs, self.actor_inputs_presence: presence
        })

    def predict_target(self, inputs,presence):
        return self.sess.run(self.target_actor_out, feed_dict={self.target_actor_inputs_dirt: inputs,  self.target_actor_inputs_presence: presence})

    def update_target_network(self):
        self.sess.run(self.update_target_network_params) 

class CriticNetwork(object):

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau

        self.critic_inputs_dirt,self.critic_inputs_presence ,self.critic_action ,self.critic_weights ,self.critic_out= self.create_critic_network('critic_network')

        self.target_critic_inputs_dirt,self.target_critic_inputs_presence,self.target_critic_action , self.target_critic_weights , self.target_critic_out = self.create_critic_network('critic_target')

        self.update_target_network_params = [self.target_critic_weights[i].assign(tf.multiply(self.critic_weights[i], self.tau) +
                                                                                 tf.multiply(self.target_critic_weights[i], 1. - self.tau))
                                                                                  for i in range(len(self.target_critic_weights))] 
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        self.loss = tf.reduce_mean(tf.square(self.predicted_q_value - self.critic_out))
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)     
        self.action_grads = tf.gradients(self.critic_out, self.critic_action)

    def create_critic_network(self,scope_name):
        with tf.variable_scope(scope_name):
            I = tf.placeholder(dtype = tf.float32, shape =[None,self.s_dim])
            P = tf.placeholder(dtype = tf.float32, shape=[None, self.s_dim ])
            A = tf.placeholder(dtype = tf.float32, shape =[None,self.a_dim])
            H1 = tf.layers.dense(I, 400, activation=tf.nn.relu,name ='H1')
            H2 = tf.layers.dense(P, 400, activation=tf.nn.relu,name ='H2')
            H3 = tf.concat((H1,H2),axis=1)
            H4= tf.layers.dense(H3,400,activation=tf.nn.relu,name ='H4') 
            H5 = tf.layers.dense(H4,300,activation=tf.nn.relu,name='H5')            
            H6 = tf.layers.dense(A,300,activation=tf.nn.relu,name='H6')
            weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)
            W1 = weights[-4]
            W2 = weights[-2]
            B = weights[-1]
            H7 = tf.nn.relu(tf.matmul(H4, W1) + tf.matmul(A, W2) + B)    
            O = tf.layers.dense(H7, 1,activation=None)
        W = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)
        return I,P,A,W,O
      
    def train(self, inputs, presence, action, predicted_q_value):
        return self.sess.run([self.critic_out, self.optimize,self.loss], feed_dict={
            self.critic_inputs_dirt: inputs,
            self.critic_inputs_presence:presence,
            self.critic_action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs,presence , action):
        return self.sess.run(self.critic_out, feed_dict={
            self.critic_inputs_dirt: inputs,
            self.critic_inputs_presence: presence,
            self.critic_action: action
        })

    def predict_target(self, inputs, presence,action):
        return self.sess.run(self.target_critic_out, feed_dict={
            self.target_critic_inputs_dirt: inputs,
            self.target_critic_inputs_presence: presence,
            self.target_critic_action: action
        })

    def action_gradients(self, inputs, presence, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.critic_inputs_dirt: inputs,
            self.critic_inputs_presence: presence,
            self.critic_action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)


class Process(object):
    
    def __init__(self,sensor='high'):
        self.sensor = sensor
    
    def get_loss_and_reward(self,summary='reward'):
        path = '/home/mtb/Documents/data/dirt_extraction_2/result/ddpg/12-21/' + self.sensor + '/ddpg_data/data/'
        # path = '/home/mtb/Documents/data/dirt_extraction_2/result/ddpg/12-21/low/ddpg_data/data/'
      
        if summary =='reward':
            self.data = pd.read_csv(path + 'reward_episode.csv')
        elif summary == 'loss':
            self.data = pd.read_csv(path + 'loss.csv')
        y_value =  self.data[self.data.columns.values[1]].values[::10]#*15829.0029756
        x_value =self.data[self.data.columns.values[0]].values[::10]
        plt.plot(x_value,y_value)
        plt.xlabel('Episodes')
        if summary =='reward':
            plt.ylabel('Rewards [Dirt/seconds]')
            plt.title( self.sensor.upper() +' level dirt reward'.upper())
        elif summary == 'loss':
            plt.ylabel('Loss')
            plt.title( self.sensor.upper() +' level dirt loss'.upper())
        plt.show()
    
    def action_file(self):
        with open(r'/home/mtb/Documents/data/dirt_extraction_2/result/ddpg/12-21/' + self.sensor + '/ddpg_data/data/action.txt') as f:
            a = f.read().splitlines()
        action_list = a[139].split(' ')
        actions = []
        for elem in action_list:
            actions.append(float(elem))
        cells = []
        for elem in actions:
            cells.append(int(elem))
        print cells

if __name__ == '__main__':
    with tf.Session() as sess:  
        state_dim = 50
        action_dim = 9       
        actor = ActorNetwork(sess, state_dim, action_dim, ACTOR_LEARNING_RATE,TAU)            
        critic = CriticNetwork(sess, state_dim, action_dim,CRITIC_LEARNING_RATE, TAU)
        DDPG().execute(sess, actor, critic, train=True)
    

