#!/usr/bin/env python
import numpy as np
import random
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf

import json

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork

from Clean import Clean
import time
import csv
import pickle
from rulo_utils.csvcreater import csvcreater
from environment_creater import EnvCreator
from rulo_utils.csvreader import csvreader
from math import exp
from random import choice, sample
from rulo_utils import csvwriter
def playGame(train_indicator=1):    #1 means Train, 0 means simply Run
    BUFFER_SIZE = 500
    BATCH_SIZE = 100
    GAMMA = 0.99
    TAU = 0.001     #Target Network HyperParameters
    LRA = 0.0001    #Learning rate for Actor
    LRC = 0.001     #Lerning rate for Critic

    action_dim = 4  #num of joints being controlled
    state_dim = 2  #num of features in state

    EXPLORE = 0.05
    episode_count = 200#210 if (train_indicator) else 1
    max_steps = 100#50 
    reward = 0
    done = False
    step = 0
    epsilon = 1 if (train_indicator) else 0.0
    indicator = 0

    #Tensorflow GPU optimization
    sess = tf.Session()
    from keras import backend as K
    K.set_session(sess)

    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer

    # Generate a Torcs environment
    EnvCreator().reset()
    env = Clean()    
    print("Now we load the weight")
    try:
        actor.model.load_weights("./actormodel.h5")
        critic.model.load_weights("./criticmodel.h5")
        actor.target_model.load_weights("./actormodel.h5")
        critic.target_model.load_weights("./criticmodel.h5")
        print("Weight load successfully")
    except:
        print("Cannot find the weight")
    
    csvcreater('/home/mtb/catkin_ws/src/rulo_pkg/rulo_rl/src/state_2017-08-03.csv',['pose'])
    csvcreater('/home/mtb/catkin_ws/src/rulo_pkg/rulo_rl/src/reward_2017-08-03.csv',['iterations','reward'])
    csvcreater('/home/mtb/catkin_ws/src/rulo_pkg/rulo_rl/src/reward_performance_2017-08-03.csv',['iterations','reward_performance'])
    reward_max_file = csvreader('/home/mtb/catkin_ws/src/rulo_pkg/rulo_rl/src/markers_2017-08-03.csv')
    value = list()

    for i in range(len(reward_max_file)):
            value.append(reward_max_file[i]['total_dust'])
    
    reward_total =float(value[1])
    print reward_total    

    iterations =0
   
    for i in range(episode_count):        
        print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))
        ob = env.reset()[0]
        if not train_indicator:
            print "start recording now"
            time.sleep(5)
        obs = [ob for ob in ob[0:2]]
        s_t = np.array(obs)       
        total_reward = 0
        for j in range(max_steps):         
            loss = 0 
            epsilon = exp(float(-50/ (2*iterations+1.0))) -1
            a_t = np.zeros([1,action_dim])
            noise_t = np.zeros([1,action_dim])
            exp_indice = np.random.random()
            if exp_indice > abs(epsilon):
                a_type = "Exploit"
                a_t = actor.model.predict(s_t.reshape(1, s_t.shape[0]))*1#rescale
            else:
                a_type = "Explore"
                a_t =np.random.uniform(0,1, size=(1,4))
                        
            action= env.step(s_t,a_t[0] )     
            ob,r_t,done = action    
         
            s_t1 = np.array(ob)     
            buff.add(s_t, a_t[0], r_t, s_t1, done)      #Add replay buffer
            
    #         #Do the batch update
            batch = buff.getBatch(BATCH_SIZE)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])  
        
            target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])             
            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] =rewards[k]            
                else:
                    y_t[k] =rewards[k] + GAMMA*target_q_values[k]                       
            
        
            if (train_indicator):
                loss += critic.model.train_on_batch([states,actions], y_t) 
                a_for_grad = actor.model.predict(states)
                grads = critic.gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()

            total_reward += r_t
            s_t = s_t1
            iterations +=1
            csvwriter('/home/mtb/catkin_ws/src/rulo_pkg/rulo_rl/src/reward_2017-08-03.csv',['iterations','reward'], {'iterations': iterations , 'reward': total_reward})               
            print("Episode", i, "Step", step, "Action", a_type, "Reward", r_t, "Loss", loss, "Epsilon", epsilon)          
        
            step += 1
            print s_t
            if done:
                # print 'done'
                print("TOTAL REWARD @ " + str(iterations) +"-th Episode  : Reward " + str(total_reward))
                print i
                reward_performance = float(total_reward)/(j+1)
                print('Performance: {0}'.format(reward_performance))
                csvwriter('/home/mtb/catkin_ws/src/rulo_pkg/rulo_rl/src/state_2017-08-03.csv',['pose'], {'pose': [0.,0.,0.]})
                csvwriter('/home/mtb/catkin_ws/src/rulo_pkg/rulo_rl/src/reward_performance_2017-08-03.csv',['iterations','reward_performance'], {'iterations': iterations , 'reward_performance': reward_performance})
                total_reward = 0
                break

        if np.mod(i, 3) == 0:
            if (train_indicator):
                print("Now we save model")
                actor.model.save_weights("./actormodel.h5", overwrite=True)
                critic.model.save_weights("./criticmodel.h5", overwrite=True)
    env.done()
    print("Finish.")
    

if __name__ == "__main__":
    playGame()
