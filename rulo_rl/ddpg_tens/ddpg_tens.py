#!/usr/bin/env python
""" 
Implementation of DDPG - Deep Deterministic Policy Gradient

Algorithm and hyperparameter details can be found here: 
    http://arxiv.org/pdf/1509.02971v2.pdf

The algorithm is tested on the Pendulum-v0 OpenAI gym task 
and developed with tflearn + Tensorflow

Author: Patrick Emami
"""
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVLEL'] = '2'
import tflearn
from replay_buffer import ReplayBuffer
from Clean import Clean
from rulo_utils.csvcreater import csvcreater
from rulo_utils.csvreader import csvread
from rulo_utils.csvwriter import csvwriter
from math import exp
# ==========================
#   Training Parameters
# ==========================
# Max training steps
MAX_EPISODES = 10
# Max episode length
MAX_EP_STEPS = 10
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.0001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.001
# Discount factor
GAMMA = 0.99
# Soft target update param
TAU = 0.001

# ===========================
#   Utility Parameters
# ===========================

# Directory for storing gym results
MONITOR_DIR = './results/gym_ddpg'
# Directory for storing tensorboard summary results
SUMMARY_DIR = '/home/mtb/rulo_ws/src/rulo_pkg/rulo_rl/ddpg_tens/summary/'
RANDOM_SEED = 5
# Size of replay buffer
BUFFER_SIZE = 10000
MINIBATCH_SIZE = 50

# ===========================
#   Actor and Critic DNNs
# ===========================


class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -2 and 2
    """

    def __init__(self, sess, state_dim, action_dim,  learning_rate, tau):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        # Actor Network
        self.inputs,self.network_params, self.out = self.create_actor_network(scope_name='actor_network')       

        # # Target Network
        self.target_inputs,self.target_network_params, self.target_out = self.create_actor_network(scope_name='actor_target')


        # Op for periodically updating target network with online network    weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # # # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # # Combine the gradients here
        self.actor_gradients = tf.gradients(
            self.out, self.network_params, -self.action_gradient)

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

    def create_actor_network(self, scope_name):

        with tf.variable_scope(scope_name) as scope:
            inputs = tf.placeholder(tf.float32, shape=[None, self.s_dim])
            hidden1 = tf.layers.dense(inputs, 400, activation= tf.nn.relu)
            hidden2 = tf.layers.dense(hidden1, 300, activation=tf.nn.relu)
            out = tf.layers.dense(hidden2, self.a_dim, activation=tf.nn.sigmoid)    
            weights =[v for v in tf.trainable_variables() if scope_name in v.name]
         
            return inputs, weights, out

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
        self.inputs, self.action, self.network_params ,self.out = self.create_critic_network(scope_name='critic_network')        
        # # # Target Network
        self.target_inputs, self.target_action, self.target_network_params ,self.target_out = self.create_critic_network(scope_name='critic_target')

        # # Op for periodically updating target network with online network weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # # Get the gradient of the net w.r.t. the action.
        # # For each action in the minibatch (i.e., for each x in xs),
        # # this will sum up the gradients of each critic output in the minibatch
        # # w.r.t. that action. Each output is independent of all
        # # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self, scope_name):

        with tf.variable_scope(scope_name) as scope:
            inputs = tf.placeholder(tf.float32, shape=[None, self.s_dim])
            action = tf.placeholder(tf.float32, shape=[None, self.a_dim])

            inp_lay1 = tf.layers.dense(inputs, 400, activation= tf.nn.relu, name='input_layer1')
            inp_lay2 = tf.layers.dense(inp_lay1, 300, activation= tf.nn.relu, name='input_layer2')       
            act_lay = tf.layers.dense(action, 300, activation= tf.nn.relu, name= 'action_layer')

            input2_kernel = [v for v in tf.trainable_variables() if v.name == scope_name + '/input_layer2/kernel:0' ][0]
            action_kernel = [v for v in tf.trainable_variables() if v.name ==scope_name +  '/action_layer/kernel:0' ][0]
            action_bias = [v for v in tf.trainable_variables() if v.name ==scope_name +  '/action_layer/bias:0' ][0]

            net = tf.nn.relu(tf.matmul(inp_lay1,input2_kernel)+ tf.matmul(action,action_kernel)+ action_bias)
            out = tf.layers.dense(net, 1)
            weights =[v for v in tf.trainable_variables() if scope_name in v.name]
            
            return inputs, action,weights, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
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

# # ===========================
# #   Tensorflow Summary Ops
# # ===========================


def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

# # ===========================
# #   Agent Training
# # ===========================


def train(sess, env, actor, critic):

    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
    saver = tf.train.Saver()
    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)

    csvcreater('/home/mtb/rulo_ws/src/rulo_pkg/rulo_rl/ddpg_tens/state_2017-08-09.csv',['pose'])
    csvcreater('/home/mtb/rulo_ws/src/rulo_pkg/rulo_rl/ddpg_tens/max_dust_2017-08-09.csv',['max_dust'])
    csvcreater('/home/mtb/rulo_ws/src/rulo_pkg/rulo_rl/ddpg_tens/reward_2017-08-09.csv',['iterations','reward'])
    csvcreater('/home/mtb/rulo_ws/src/rulo_pkg/rulo_rl/ddpg_tens/reward_performance_2017-08-09.csv',['iterations','reward_performance'])

    env.create_env()
    # reward_max_file = csvread('/home/mtb/rulo_ws/src/rulo_pkg/rulo_rl/ddpg_tens/markers_2017-08-09.csv')
    # value = list()

    # for i in range(len(reward_max_file)):
    #         value.append(reward_max_file[i]['total_dust'])
    
    # reward_total =float(value[1])
    # print reward_total
    
    iterations =0
    clean = 0
    for i in xrange(MAX_EPISODES):               

        ep_reward = 0
        ep_ave_max_q = 0
        ob = env.reset()[0]
        obs = [ob for ob in ob[0:2]]
        s_t = np.array(obs, ndmin=2)
        total_reward = 0
      
        for j in xrange(MAX_EP_STEPS):
       
            epsilon =exp(float(-50/ (2*iterations+1.0))) -1
            a_t = np.zeros([1,1])            
            exp_indice = np.random.random()
         
            if exp_indice > abs(epsilon):
                a_type = "Exploit"
                a_t =  actor.predict(np.reshape(s_t, (1, actor.s_dim)))#actor.model.predict(s_t.reshape(1, s_t.shape[0]))*1#rescale
            else:
                a_type = "Explore"
                a_t =np.random.uniform(0,1, size=(1,actor.a_dim))              
           
            print 'state: ' +str(s_t[0])
            # env.step(s_t[0],a_t[0] )  
        
            s_t1,r_t,done = env.step(s_t[0],a_t[0] )  
  
            
            replay_buffer.add(np.reshape(s_t, (2,)), np.reshape(a_t, (1,)), r_t,
                              done, s_t1  )

# # #             # Keep adding experience to the memory until
# # #             # there are at least minibatch size samples
            if replay_buffer.size() > MINIBATCH_SIZE:
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(MINIBATCH_SIZE)

            # #     # Calculate targets
                
              
                target_q = critic.predict_target(
                    s2_batch, actor.predict_target(s2_batch))
                
                
                y_i = []
                for k in xrange(MINIBATCH_SIZE):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + GAMMA * target_q[k])

#                 # Update the critic given the targets
                predicted_q_value, _ = critic.train(
                    s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))

                ep_ave_max_q += np.amax(predicted_q_value)

             # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

            s_t =np.reshape(s_t1, (1,2))
            ep_reward += r_t
            print 'next_state: '  + str(s_t1)
           
            iterations +=1
            print ('Step {0} ,  '.format(iterations) + ' Reward: {0}'.format(str(r_t) ))
            
            if (iterations //5 ==0):
                saver.save(sess,'/home/mtb/rulo_ws/src/rulo_pkg/rulo_rl/ddpg_tens/model/model_2017_08_09.ckpt' )
                print 'saving'
            if done:

                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: ep_reward,
                    summary_vars[1]: ep_ave_max_q 
                })

                writer.add_summary(summary_str, i)
                writer.flush()
                print ('Episode:  {0} ,  '.format(i+1) + 'Total  Reward: {0} , '.format(ep_reward) + 'Qmax: {}'.format(ep_ave_max_q))
                
                print 'done'

                break


def main(_):
    with tf.Session() as sess:
      
        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)
        
        state_dim =  2
        action_dim = 1
        actor = ActorNetwork(sess, state_dim, action_dim, 
                             ACTOR_LEARNING_RATE, TAU)

        critic = CriticNetwork(sess, state_dim, action_dim,
                            CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars())
        env = Clean()

        train(sess, env, actor, critic)


if __name__ == '__main__':
    tf.app.run()
