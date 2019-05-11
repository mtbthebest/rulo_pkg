#!/usr/bin/env	python
import rospy
import os
import sys
import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from rulo_base.markers import VizualMark, TextMarker
import pandas as pd
from pandas import DataFrame, Series
from Buffer import Memory
from rulo_utils.csvwriter import csvwriter
import random
from time import sleep

BUFFER_SIZE = 1000
MINIBATCH_SIZE = 100
RANDOM_SEED = 1234
GAMMA_FACTOR = 0.95
grid_size = 0.25
MAX_EPISODES = 1000 # 100000000
learning_rate = 0.001
TAU = 0.0001
STEP = 49
SUM_DIR = '/home/mtb/rulo_ws/src/rulo_pkg/rulo_ml/dqn/summary/'


class DQN(object):
    def __init__(self, sess, state_dim, action_dim):
        self.data = pd.read_csv(
            '/home/mtb/rulo_ws/src/rulo_pkg/rulo_ml/dqn/data/dqn_data_state_dirt_300.csv', index_col=[0])
        self.all_actions = self.rewrite(columns='actions')
        # self.all_rewards = self.rewrite(columns='rewards', type='float')
        self.rewards = pd.read_csv('/home/mtb/rulo_ws/src/rulo_pkg/rulo_ml/data/rewards_ddpg_300.csv', index_col=[0])
        self.all_next_state = self.rewrite(columns='next_state', type='float')
        self.start = list(self.data['start'].values)
        self.positions = self.rewrite(columns='positions', type='float')
        self.possibility = self.rewrite(columns='possibility', type='float')

        self.states = list(self.data.iloc[:].index.values)
        self.max_val = max(self.states)
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.q_dirt_inputs,self.q_presence_inputs ,self.q_weights, self.q_outputs = self.create_network('q_net')
        self.q_targets_dirt_inputs, self.q_targets_presence_inputs,self.q_targets_weights, self.q_targets_outputs, = self.create_network('q_net_targ')
        self.update_target_params = [self.q_targets_weights[i].assign(self.q_weights[i]) for i in range(len(self.q_targets_weights))]
        self.predicted_q = tf.placeholder(tf.float32, [None, self.action_dim])
        self.loss = tf.reduce_mean(tf.square(self.predicted_q - self.q_outputs))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

    def create_network(self, scope_name):
        with tf.variable_scope(scope_name):
            I = tf.placeholder(tf.float32, shape=[None, self.state_dim ])
            P = tf.placeholder(tf.float32, shape=[None, self.state_dim ])
            H1 = tf.layers.dense(I, 400, activation=tf.nn.relu)
            H2 = tf.layers.dense(P, 400, activation=tf.nn.relu)
            H3 = tf.concat((H1,H2),axis=1)
            H4 = tf.layers.dense(H3,300,activation=tf.nn.relu)
            O = tf.layers.dense(H4,self.action_dim,activation=None)
        W = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)    
        return I , P , W , O
       
    def train(self, inputs, presence,predicted_q):
        return self.sess.run([self.optimizer, self.loss, self.q_outputs], {self.q_dirt_inputs: inputs,self.q_presence_inputs:presence ,self.predicted_q: predicted_q})

    def predict(self, inputs,presence):
        return sess.run(self.q_outputs, {self.q_dirt_inputs: inputs ,self.q_presence_inputs: presence })

    def predict_target(self, inputs,presence):
        return sess.run(self.q_targets_outputs, {self.q_targets_dirt_inputs: inputs , self.q_targets_presence_inputs:presence })

    def update_weights(self):
        return sess.run(self.update_target_params)

    def save_model(self):
        self.saver.save(
            self.sess, '/home/mtb/rulo_ws/src/rulo_pkg/rulo_ml/dqn/model/var.ckpt')

    def restore(self):
        if os.path.isfile('/home/mtb/rulo_ws/src/rulo_pkg/rulo_ml/dqn/model/checkpoint'):
            self.saver.restore(
                sess, '/home/mtb/rulo_ws/src/rulo_pkg/rulo_ml/dqn/model/var.ckpt')
            print 'Restored'

    def summary(self,episodes,explore,cost_function,episode_reward,wander,cells, total_rewards):
        csvwriter(SUM_DIR + 'summary.csv',
                  ['episodes', 'explore', 'cost_function','episode_reward','wander','cells','total_rewards'], [[episodes],[explore],[cost_function],[episode_reward],[wander],[cells], [total_rewards]])

    def run(self):
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.restore()
        iterations = 0
        self.replay_buffer = Memory(BUFFER_SIZE, RANDOM_SEED)
        self.update_weights()
        explore = 0
        exploit = 0
        epsilon = 0.5
        # self.rewards = self.all_rewards[:]
        self.input_states = []
        for i in range(len(self.states)):
            cell_val = self.states[i] / max(self.states)
            # if cell_val >= 1.0:
            #     color = 0.0
            # else:
            #     color = round(((255.0 - cell_val * 255.0) / 255.0), 4)
            self.input_states.append([ cell_val,0.0])

        self.bad_actions = OrderedDict()        
        for episodes in range(MAX_EPISODES):
            
            self.all_states = self.input_states[:]
            start_cell = 144  # self.get_state()
            self.all_states[start_cell][1] = 1.0
            s_t_d, s_t_p = np.array(self.all_states)[:,0 ], np.array(self.all_states)[:,1 ]
           
            last_state = start_cell
            initial_state = [s_t_d,s_t_p]
            
            terminal = False
            episode_terminal = False
            
            self.tested = OrderedDict()
            self.wandered = []
            self.cost_function_list = np.array([])           

            self.simulate(s_t_d,s_t_p, last_state, episode_terminal,  episodes, initial_state)

    def simulate(self, s_t_d,s_t_p, last_state, episode_terminal, episodes, initial_state):
        self.explore = 0
        total_rewards = 0.0
        terminal = False
        go_explore = False        
        for j in range(300):             
            action = self.get_operation(s_t_d,s_t_p,j, episodes, last_state, exploration=False)            
            s_t1_d, s_t1_p, reward, new_position, terminal = self.do_action(last_state, s_t_d, s_t_p, action)  
            # print last_state,action ,new_position, reward    
            if new_position not in self.wandered:
                self.wandered.append(new_position)
            try:
                if action not in self.tested[last_state]:
                    self.tested[last_state].append(action)
            except:
                self.tested[last_state] = [action]  

                 
            total_rewards += reward                                
            self.replay_buffer.add(s_t_d, s_t_p ,action, reward, terminal, s_t1_d, s_t1_p)          
            last_state = new_position
            s_t_d , s_t_p= s_t1_d , s_t1_p         
        cost_function  = self.learn()
        self.save_model()    
        self.wandered = []
        cells, episode_reward = self.get_reward(initial_state, present_position=144)
        try:
            print episodes,  total_rewards,self.explore , cost_function
            self.summary(episodes,self.explore,cost_function,episode_reward,len(self.wandered),cells, total_rewards)
        except:
            pass  

    def do_action(self, present_position, s_t_d, s_t_p, action):
        new_state_d = np.copy(s_t_d)
        new_state_p = np.copy(s_t_p)
        new_position = int(self.positions[present_position][action])
        if self.possibility[present_position][action] == 1.0:
            if new_position not in self.wandered:
                # reward = self.rewards[present_position][action]
                # print reward
                reward = self.rewards[str(present_position)][new_position]
            else:
                reward = -10.0
            new_state_d[new_position] = 0.0
            new_state_p[present_position] = 0.0
            new_state_p[new_position] = 1.0
            terminal = False
        else:
            reward = -100.0
            terminal = True
       
        return new_state_d, new_state_p, reward, new_position, terminal

    def get_state(self):
        return np.random.randint(0, STEP + 1, 1)[0]

    def rewrite(self, columns='action', type='int'):
        res = []
        for elem1 in self.data[columns].values:
            if type == 'int':
                trans = map(int, elem1[1:-1].split(','))
                res.append(trans)
            if type == 'float':
                frac = elem1[1:-1].split(',')
                trans = []
                for val in frac:
                    try:
                        trans.append(float(val))
                    except:
                        pass
                res.append(trans)
        return res

    def get_operation(self, s_t_d,s_t_p, step, episodes, last_state, exploration=False):
        self.epsilon = 1- np.exp(-float(step) / (float(episodes) +1))
        action_list = []
        if not exploration:
            q = self.predict(s_t_d.reshape(1, self.state_dim),s_t_p.reshape(1, self.state_dim))
            action = np.argmax(q)
            if np.random.random() > self.epsilon:
                a_type = "Exploit"
                # exploit += 1
            else:
                a_type = "Explore"
                self.explore += 1
                try:
                    skip_action = [action] + self.tested[last_state]
                except:
                    skip_action = [action]
                for m in range(self.action_dim):
                    if m not in skip_action:
                        action_list.append(m)
                if action_list:
                    action = np.random.choice(action_list, 1)[0]
       
        return action        

    def learn(self):
        if self.replay_buffer.size() >= BUFFER_SIZE:
            s_batch_d, s_batch_p,a_batch, r_batch, t_batch, s2_batch_d, s2_batch_p = self.replay_buffer.sample_batch(MINIBATCH_SIZE)
            
            target_q = self.predict_target(
                np.array(s2_batch_d).reshape(BUFFER_SIZE, self.state_dim),np.array(s2_batch_p).reshape(BUFFER_SIZE, self.state_dim))
          
            q_val = self.predict(np.array(s_batch_d).reshape([BUFFER_SIZE, self.state_dim]),np.array(s_batch_p).reshape([BUFFER_SIZE, self.state_dim]))
            
            for k in range(BUFFER_SIZE):
                if t_batch[k]:
                    q_val[k][np.int(a_batch[k])] = r_batch[k]
                else:
                    q_val[k][np.int(a_batch[k])] = r_batch[k] + GAMMA_FACTOR * np.amax(target_q[k])
             
            for m in range(int(BUFFER_SIZE / MINIBATCH_SIZE)):
                _, loss, _ = self.train(np.array(s_batch_d[m * MINIBATCH_SIZE: (m + 1) * MINIBATCH_SIZE]).reshape([MINIBATCH_SIZE, self.state_dim]),
                                        np.array(s_batch_p[m * MINIBATCH_SIZE: (m + 1) * MINIBATCH_SIZE]).reshape([MINIBATCH_SIZE, self.state_dim]),
                                        q_val[m * MINIBATCH_SIZE: (m + 1) * MINIBATCH_SIZE])
                self.cost_function_list = np.concatenate((self.cost_function_list, [loss]))

        self.update_weights()
        return np.mean(self.cost_function_list)

    def get_reward(self, state, present_position=0):        
        state_d , state_p =state[0], state[1]
        terminal = False
        all_reward = 0.0
        result = 0
        all_cells =[]
        while not terminal:
            q = self.predict(state_d.reshape(1, self.state_dim),state_p.reshape(1, self.state_dim))
            # if result == 0:
            #     print q
            action = np.argmax(q)
            s_t1_d, s_t1_p, reward, new_position, terminal = self.do_action(present_position, state_d , state_p, action)
            if new_position not  in all_cells:
                all_cells.append(new_position)
                all_reward +=reward           
            else:
                break
            if not terminal:
                state_d , state_p, present_position = s_t1_d, s_t1_p, new_position
            else:
                break
        return all_cells,all_reward




if __name__ == '__main__':
   with tf.Session() as sess:
       state_dim = 300
       action_dim = 9
       DQN(sess, state_dim, action_dim).run()
