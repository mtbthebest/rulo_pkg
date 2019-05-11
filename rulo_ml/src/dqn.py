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
from Memory import Memory
from rulo_utils.csvwriter import csvwriter
import random

BUFFER_SIZE = 5000
MINIBATCH_SIZE = 500
RANDOM_SEED = 1234
GAMMA_FACTOR = 0.95
grid_size = 0.25
MAX_EPISODES = 1000  # 100000000
learning_rate = 0.001
TAU = 0.0001
STEP = 49
SUM_DIR = '/home/mtb/rulo_ws/src/rulo_pkg/rulo_ml/dqn/summary/'


class DQN(object):
    def __init__(self, sess, state_dim, action_dim):
        self.data = pd.read_csv(
            '/home/mtb/rulo_ws/src/rulo_pkg/rulo_ml/dqn/data/dqn_data_state_dirt.csv', index_col=[0])
        self.all_actions = self.rewrite(columns='actions')
        self.all_rewards = self.rewrite(columns='rewards', type='float')
        self.all_next_state = self.rewrite(columns='next_state', type='float')
        self.start = list(self.data['start'].values)
        self.positions = self.rewrite(columns='positions', type='float')
        self.possibility = self.rewrite(columns='possibility', type='float')
        # print self.possibility
        self.states = list(self.data.iloc[:].index.values)
        self.max_val = max(self.states)
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.q_inputs, self.q_weights, self.q_outputs = self.create_network('q_net')
        self.q_targets_inputs, self.q_targets_weights, self.q_targets_outputs = self.create_network('q_net_targ')
        self.update_target_params = [self.q_targets_weights[i].assign(self.q_weights[i]) for i in range(len(self.q_targets_weights))]
        self.predicted_q = tf.placeholder(tf.float32, [None, self.action_dim])
        self.loss = tf.reduce_mean(tf.square(self.predicted_q - self.q_outputs))

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self.loss)

    def create_network(self, scope_name):
        with tf.variable_scope(scope_name):
            I = tf.placeholder(tf.float32, shape=[None, self.state_dim + 1])
            H1 = tf.layers.dense(I, 400, activation=tf.nn.relu)
            H2 = tf.layers.dense(H1, 300, activation=tf.nn.relu)
            O = tf.layers.dense(H2, self.action_dim, activation=None)
        W = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name)
        return I, W, O

    def train(self, inputs, predicted_q):
        return self.sess.run([self.optimizer, self.loss, self.q_outputs], {self.q_inputs: inputs, self.predicted_q: predicted_q})

    def predict(self, inputs):
        return sess.run(self.q_outputs, {self.q_inputs: inputs})

    def predict_target(self, inputs):
        return sess.run(self.q_targets_outputs, {self.q_targets_inputs: inputs})

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

    def summary(self, episodes, loss, rewards, epsilon):
        csvwriter(SUM_DIR + 'summary.csv', ['episodes', 'loss', 'rewards', 'epsilon'], [
                  [episodes], [loss], [rewards], [epsilon]])

    def run(self):
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.restore()
        iterations = 0
        replay_buffer = Memory(BUFFER_SIZE, RANDOM_SEED)
        self.update_weights()
        explore = 0
        exploit = 0
        epsilon = 0.8
        self.rewards = self.all_rewards[:]

        for episodes in range(MAX_EPISODES):
            self.all_states = [self.states[i] /
                               max(self.states) for i in range(len(self.states))]
            start_cell = self.get_state()
            s_t = np.concatenate(
                ([start_cell], self.all_states)).reshape(1, 51)
            q = self.predict(s_t)
            self.last_state = start_cell
            self.tested_position = []
            self.rewarded = {'good': [], 'bad': []}
            terminal = False
            # self.rewards = [float(self.states[i]) * 10.0/ sum(self.states) for i in range(len(self.states))]
            # total_rewards = self.rewards[start_cell]
            # float(self.states[start_cell]) * 10.0 / sum(self.states)
            total_rewards = 0.0
            # self.rewards = self.all_rewards[:]
            # self.rewards[start_cell] = 0.0
            # explore =0
            step = 0
            self.tested = OrderedDict()
            self.wander_cells = []
            # all_state = []
            # print self.rewards
            while not terminal:
                # for l in range(30):
                if epsilon < 0.000000001:
                    epsilon = 0.9 - 0.8 * float(episodes) / 10000.0
                try:
                    epsilon -= 0.3 / np.abs(10000.0 - step)
                except:
                    epsilon -= 0.3 / float(step)
                iterations += 1
                q = self.predict(s_t)
                action = np.argmax(q)
                if np.random.random() > epsilon:
                    a_type = "Exploit"
                    exploit += 1
                else:
                    a_type = "Explore"
                    explore += 1
                    action_list = []
                    try:
                        skip_action = [action] + self.tested[self.last_state]
                    except:
                        skip_action = [action]
                    for m in range(8):
                        if m not in skip_action:
                            action_list.append(m)
                    if action_list:
                        action = np.random.choice(action_list, 1)[0]
                # print 'Episode: %d , Iter: %d , Explore: %d , Exploit: %d , All_state: %d' % (episodes, iterations, explore, exploit,len(all_state))
                s_t1, reward, new_position = self.do_action(
                    self.last_state, s_t[0], action)
                # print self.last_state , action , new_position, reward
                try:
                    self.tested[self.last_state].append(action)
                except:
                    self.tested[self.last_state] = [action]
                if new_position not in self.tested_position:
                    self.tested_position.append(new_position)
                total_rewards += reward

                replay_buffer.add(s_t[0], action, reward, terminal, s_t1)
                step += 1
                if replay_buffer.size() >= BUFFER_SIZE:
                    s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(
                        MINIBATCH_SIZE)
                    target_q = self.predict_target(
                        np.array(s2_batch).reshape([BUFFER_SIZE, self.state_dim + 1]))
                    q_val = self.predict(np.array(s_batch).reshape(
                        [BUFFER_SIZE, self.state_dim + 1]))
                    for k in range(BUFFER_SIZE):
                        q_val[k][np.int(a_batch[k])] = r_batch[k] + \
                            GAMMA_FACTOR * np.amax(target_q[k])

                    for m in range(int(BUFFER_SIZE / MINIBATCH_SIZE)):
                        _, loss, _ = self.train(np.array(s_batch[m * MINIBATCH_SIZE: (m + 1) * MINIBATCH_SIZE]).reshape(
                            [MINIBATCH_SIZE, self.state_dim + 1]), q_val[m * MINIBATCH_SIZE: (m + 1) * MINIBATCH_SIZE])
                self.update_weights()
                if len(self.tested_position) == 50 or step >= 10000:
                    terminal = True
                    break
                self.last_state = new_position
                s_t = s_t1.reshape(1, self.state_dim + 1)
                # print
                # print self.tested_position ,  total_rewards , step
            self.save_model()
            try:
                print episodes, loss, total_rewards
                self.summary(episodes, loss, total_rewards, epsilon)
            except:
                pass

    def carry_out(self, state, s_t, action):
        index = self.states.index(state)
        next_state = self.all_next_state[index][action]
        s_t1 = next_state / 1000.0
        reward = self.rewards[index][action]
        # self.rewards[index][action] = 0.0
        return next_state, s_t1, reward

    def make_action(self, present_position, state, action):
        new_state = state[:]
        new_position = self.positions[present_position][action]
        new_state[1:][int(new_position)] = 0.0
        new_state[0] = new_position
        reward = self.rewards[int(new_position)]
        self.rewards[int(new_position)] = 0.0
        return new_state, reward, int(new_position)

    def do_action(self, present_position, state, action):
        new_state = state[:]
        new_position = self.positions[present_position][action]
        new_state[1:][int(new_position)] = 0.0
        new_state[0] = new_position
        reward = self.rewards[present_position][action]
        if [present_position, action] in self.wander_cells:
            if self.possibility[present_position][action] == -1.0:
                reward = -100.0
            else:
                reward = 0.0
        else:
            if int(new_position) in self.tested_position:
                if self.rewards[present_position][action] == -100.0:
                    reward = -100.0
                elif self.rewards[present_position][action] >= 0.0:
                    if int(new_position) in self.rewarded['good']:
                        reward = 0.0
                    else:
                        reward = self.rewards[present_position][action]
                        self.rewarded['good'].append(int(new_position))
                        self.wander_cells.append([present_position, action])
            else:
                reward = self.rewards[present_position][action]
                if reward == -100.0:
                    self.rewarded['bad'].append(int(new_position))
                else:
                    self.rewarded['good'].append(int(new_position))
                self.wander_cells.append([present_position, action])
        # print reward
        return new_state, reward, int(new_position)

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


class Rewards(object):
    def __init__(self):
        self.states = np.load(
            '/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/data/pose.npy')
        self.dirt_lev = np.load(
            '/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/data/dirt.npy')

    def get_near_cells(self):
        states = OrderedDict()
        limit = 50
        for i in range(limit):
            states[i] = OrderedDict()
            position = self.states[i]
            states[i][0] = [position[0] + grid_size, position[1]]
            states[i][1] = [position[0] + grid_size, position[1] + grid_size]
            states[i][2] = [position[0], position[1] + grid_size]
            states[i][3] = [position[0] - grid_size, position[1] + grid_size]
            states[i][4] = [position[0] - grid_size, position[1]]
            states[i][5] = [position[0] - grid_size, position[1] - grid_size]
            states[i][6] = [position[0], position[1] - grid_size]
            states[i][7] = [position[0] + grid_size, position[1] - grid_size]
        rospy.init_node('path')
        poses = states[28].values()
        VizualMark().publish_marker(pose=[list(self.states[28])], sizes=[
            [0.25, 0.25, 0.0]] * 1, color=['Default'], id_list=[0], publish_num=50)
        TextMarker().publish_marker(
            text_list=states[28].keys(), pose=states[28].values())
        VizualMark().publish_marker(poses, sizes=[
            [0.25, 0.25, 0.0]] * 8, color=['Green', 'Blue', 'Yellow', 'Aqua', 'Gray', 'Pink', 'Orange', 'Maroon'], id_list=range(1, 9), publish_num=2)
        # max_dirt = np.amax(self.dirt_lev)
        # dirt_level = sorted(list(self.dirt_lev[:limit]),reverse=True)

        # mydata =[]
        # for i in range(limit):
        #     data = OrderedDict()
        #     if self.dirt_lev[i] !=0.0:
        #         data[self.dirt_lev[i]] = OrderedDict()
        #         state = states[i]
        #         for keys, values in state.items():
        #             rewarded = False
        #             for j in range(limit):
        #                 if values[0] ==self.states[j][0] and values[1] == self.states[j][1] and self.dirt_lev[j] >0.0:
        #                     rewarded = True
        #                     if float(self.dirt_lev[j]) > 0.0:
        #                         r = float(self.dirt_lev[j]) *100.0 / sum(dirt_level)

        #                     data[self.dirt_lev[i]][keys] = [1.0,self.dirt_lev[j], r]
        #                     # break
        #             if not rewarded:
        #                 data[self.dirt_lev[i]][keys] = [ -1.0,self.dirt_lev[i],0.0 ]

        #         a = data[self.dirt_lev[i]].keys()
        #         b = data[self.dirt_lev[i]].values()
        #         actions = list(np.array(a))
        #         next_state = list(np.array(b)[:,1])
        #         rewards = list(np.array(b)[:, 2])
        #         d = DataFrame(
        #             {'actions':[actions]}, index=[self.dirt_lev[i]])
        #         e = DataFrame(
        #             {'next_state': [next_state]}, index=[self.dirt_lev[i]])
        #         f = DataFrame(
        #             {'rewards': [rewards]}, index=[self.dirt_lev[i]])
        #         res = pd.concat([d,e,f],axis=1)
        #         mydata.append(res)
        # else:
        #     pass
        # c = pd.concat(mydata)
        # c.to_csv('/home/mtb/rulo_ws/src/rulo_pkg/rulo_ml/dqn/data/dqn_data.csv')


class RewardState(object):
    def __init__(self):
        self.states = np.load(
            '/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/data/pose.npy')
        self.dirt_lev = np.load(
            '/home/mtb/rulo_ws/src/rulo_pkg/rulo_base/data/dirt.npy')

    def get_near_cells(self):
        myreward = []
        states = OrderedDict()
        limit = 50
        for i in range(limit):
            states[i] = OrderedDict()
            position = self.states[i]
            states[i][0] = [position[0] + grid_size, position[1]]
            states[i][1] = [position[0] + grid_size, position[1] + grid_size]
            states[i][2] = [position[0], position[1] + grid_size]
            states[i][3] = [position[0] - grid_size, position[1] + grid_size]
            states[i][4] = [position[0] - grid_size, position[1]]
            states[i][5] = [position[0] - grid_size, position[1] - grid_size]
            states[i][6] = [position[0], position[1] - grid_size]
            states[i][7] = [position[0] + grid_size, position[1] - grid_size]
            states[i][8] = [position[0], position[1]]

        max_dirt = np.amax(self.dirt_lev)
        dirt_level = sorted(list(self.dirt_lev[:limit]), reverse=True)
        mydata = []
        for i in range(limit):
            data = OrderedDict()
            # if self.dirt_lev[i] !=0.0:
            data[self.dirt_lev[i]] = OrderedDict()
            state = states[i]
            for keys, values in state.items():
                rewarded = False
                for j in range(limit):
                    if values[0] == self.states[j][0] and values[1] == self.states[j][1]:
                        rewarded = True
                        if float(self.dirt_lev[j]) > 0.0:
                            r = float(self.dirt_lev[j]) * \
                                2000 / sum(dirt_level)
                        else:
                            r = 0.0
                        myreward.append(r)
                        data[self.dirt_lev[i]][keys] = [
                            1.0, self.dirt_lev[j], r, i, j]
                        # break
                if not rewarded:
                    data[self.dirt_lev[i]][keys] = [-1.0,
                                                    self.dirt_lev[i], -100.0, i, i]

            a = data[self.dirt_lev[i]].keys()
            b = data[self.dirt_lev[i]].values()
            action_possibility = list(np.array(b)[:, 0])
            actions = list(np.array(a))
            next_state = list(np.array(b)[:, 1])
            rewards = list(np.array(b)[:, 2])
            start = list(np.array(b)[:, 3])
            positions = list(np.array(b)[:, 4])
            d = DataFrame(
                {'actions': [actions]}, index=[self.dirt_lev[i]])
            e = DataFrame(
                {'next_state': [next_state]}, index=[self.dirt_lev[i]])
            f = DataFrame(
                {'rewards': [rewards]}, index=[self.dirt_lev[i]])
            g = DataFrame(
                {'positions': [positions]}, index=[self.dirt_lev[i]])
            h = DataFrame(
                {'start': i}, index=[self.dirt_lev[i]])

            n = DataFrame(
                {'possibility': [action_possibility]}, index=[self.dirt_lev[i]])
            res = pd.concat([d, e, h, g, f, n], axis=1)
            mydata.append(res)
        else:
            pass
        c = pd.concat(mydata)
        # print sum(myreward)
        c.to_csv(
            '/home/mtb/rulo_ws/src/rulo_pkg/rulo_ml/dqn/data/dqn_data_state_dirt.csv')


class Performance:
    def __init__(self):
        self.data = pd.read_csv(
            '/home/mtb/rulo_ws/src/rulo_pkg/rulo_ml/dqn_server/01-26/summary.csv')

        plt.plot(self.data['episodes'][::500], self.data['loss'][::500])
        plt.xlabel('Iterations')
        plt.ylabel('loss')
        plt.title('Total loss in each iteration')
        plt.show()

# if __name__ == '__main__':
#    with tf.Session() as sess:
#        state_dim = 50
#        action_dim = 9
#        DQN(sess,state_dim, action_dim ).run()

    RewardState().get_near_cells()

    # a = [4,5,6,7]
    # b =[4,5,6,7,8,9]
    # c =[3,2]
    # d = np.array([[a,
    #             b,
    #             c]])
    # print d
    # a = np.random.choice([2,4,5,6],1)
    # print a
    # a = [5,4,5,6,7,25,6,96]
    # b = np.array(a)
    # print b[[7,0,1,3,4,2,7]]
    # np.random.shuffle(a)
    # print a
