#!/usr/bin/env	python
import os
import sys
import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer
import numpy as np
import pandas as pd
from rulo_utils.graph_plot import Plot
import matplotlib.pyplot as plt
from rulo_utils.csvwriter import csvwriter
from pandas import Series, DataFrame
from collections import OrderedDict
from Wander import Wander
from ReplayBuffer import ReplayBuffer 
import tflearn
RANDOM_SEED = 1234
BUFFER_SIZE = 10000
MINIBATCH_SIZE = 500

EPISODES = 1000000
STEPS = 500
ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.001
EXPLORE = 10000.0
GAMMA_FACTOR = 0.99

TAU = 0.001

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
        # saver = tf.train.Saver()        
        # if os.path.isfile('/home/mtb/rulo_ws/src/rulo_pkg/rulo_ml/ddpg/dlbox/01-21-20:40/ddpg_data/model/checkpoint'):
        #     saver.restore(
        #         sess, '/home/mtb/rulo_ws/src/rulo_pkg/rulo_ml/ddpg/dlbox/01-21-20:40/ddpg_data/model/var.ckpt')
        #     print 'Restored'    
        env = Wander()
        iterations = 0     
        # exploit =0
        # exploire =0   
        if train: 
            actor.update_target_network()  
            critic.update_target_network()
            replaybuffer=ReplayBuffer(BUFFER_SIZE,RANDOM_SEED)            
            iterations = 0    
            explore =0
            exploit =0
            for i in range(EPISODES):                
                s_t= env.initialize()                           
                total_reward = env.save(action=24)
                terminal = False                
                for j in range(STEPS):
                    # if i %300 == 0:
                    #     iterations =0
                    epsilon = 100.0 * np.exp(-float(iterations) / 100000.0)
                    iterations +=1
                    if np.random.random() > epsilon:
                        a_type = "Exploit"
                        exploit +=1
                        # a = actor.predict(s_t)
                    else:
                        a_type = "Explore"
                        explore +=1
                        # act = np.random.randint(0, 50, 1)[0]
                        # a = np.zeros([1, 50], dtype=float)
                        # a[0][act] = 1.0
                        
                    print 'Episode: %d , Iter: %d , Explore: %d , Exploit: %d'%(i,iterations,explore, exploit)

                #     s_t1,r,action,update,terminal = env.step(s_t,a)                     
                #     total_reward +=r                  
                #     if update: 
                #         replaybuffer.add(s_t[0], a[0],r,terminal, s_t1[0])                    
                #     if replaybuffer.size() > MINIBATCH_SIZE:
                #         s_t_batch, a_batch, r_batch, t_batch, s_t1_batch = \
                #             replaybuffer.sample_batch(MINIBATCH_SIZE)   
                #         target_q = critic.predict_target(s_t1_batch, actor.predict_target(s_t1_batch))
                #         y_i = []
                #         for k in xrange(MINIBATCH_SIZE):
                #             if t_batch[k]:
                #                 y_i.append(r_batch[k])
                #             else:
                #                 y_i.append(r_batch[k] + GAMMA_FACTOR * target_q[k])                        
                #         predicted_q_value =critic.train(s_t_batch, a_batch, np.reshape(y_i,(MINIBATCH_SIZE,1)))
                #         loss = predicted_q_value[2]
                #         # print loss
                #         a_outs = actor.predict(s_t_batch)
                #         grads = critic.action_gradients(s_t_batch, a_outs)
                #         actor.train(s_t_batch, grads[0])
                #         actor.update_target_network()
                #         critic.update_target_network()
                #     s_t = s_t1

                #     if terminal:
                #         # print 'Episode %d , Reward: %.6f  ' % (i, total_reward)
                #         break
                # env.save(save=True, episode=i)
                # # saver.save(sess, '/home/mtb/rulo_ws/src/rulo_pkg/rulo_ml/model/var.ckpt')   
                # try:
                #     env.save_loss(loss,i)
                # except:
                #     pass
                # try:
                #     print 'Episode %d , Reward: %.6f , Loss: %.7f '%(i,total_reward, loss)
                # except:
                #     pass

        else:
            self.states = np.load('/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/data/pose.npy')
            s_t = self.states[144].reshape([1, 2])
            self.all_action = []
            for t in range(100):
                # s_t,num = env.reset()
                a = actor.predict(s_t)
                if np.argmax(a) not in self.all_action:
                    s_t = self.states[np.argmax(a)].reshape(1, 2)
                    self.all_action.append(np.argmax(a))
                s_t = self.states[np.argmax(a)].reshape(1, 2)
            print self.all_action
                # self.action = [num]
                # for i in range(100):
                #     a = actor.predict(s_t)
                #     if np.argmax(a) not in self.action:
                #         self.action.append(np.argmax(a))
                #         s_t = self.states[np.argmax(a)].reshape(1,2)
                #     else:
                #         break
                # self.all_action.append(self.action)
            # print self.all_action
            # dataframe= csvwriter('/home/mtb/Desktop/ddpg_21.low.csv', headers=['actions'],
            #                      rows = [self.all_action])
            # self.action = OrderedDict()     
            # for j in range(623):
            #     self.action[j] = [j]
            #     for m in range(100):
            #         s_t = self.states[j].reshape([1,2]) 
            #         a = actor.predict(s_t)                
            #         action= np.argmax(a)                    
            #         if action not in self.action[j]:
            #             s_t = np.reshape(self.states[action],(1,2))
            #             self.action[j].append(action)
            #     print self.action[j]        
            
            # dataframe= csvwriter('/home/mtb/Desktop/ddpg_21.low.csv', headers=['cells', 'actions'],
            #                      rows = [self.action.keys(), self.action.values()])
            # print self.action


class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.
    """

    def __init__(self, sess, state_dim, action_dim,learning_rate, tau):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau

        # Actor Network
        self.inputs, self.out= self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[
            len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.actor_gradients = tf.gradients(
            self.out, self.network_params, -self.action_gradient)

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.fully_connected(inputs, 400, activation='relu')
        net = tflearn.fully_connected(net, 300, activation='relu')
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(
            net, self.a_dim, activation='tanh', weights_init=w_init)
       
        return inputs, out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars
    

class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables(
        )[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        net = tflearn.fully_connected(inputs, 400, activation='relu')

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, 300)
        t2 = tflearn.fully_connected(action, 300)

        net = tflearn.activation(
            tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, activation='tanh' ,weights_init=w_init)
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize,self.loss], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
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
        state_dim = 2
        action_dim = 50
        # for i in range(10):
        actor = ActorNetwork(sess, state_dim, action_dim, ACTOR_LEARNING_RATE,TAU)
        # print tf.trainable_variables()
        critic = CriticNetwork(sess, state_dim, action_dim,
                               CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars())
        DDPG().execute(sess, actor, critic, train=True)
    
    # Process('high').action_file()
    # Process('low').get_loss_and_reward(summary='loss')
