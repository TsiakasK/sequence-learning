#!/usr/bin/python
import numpy as np
import random
import argparse
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
import json
from keras import backend as K
import toy_environment as env
from actor import ActorNetwork
from critic import CriticNetwork
import matplotlib.pyplot as plt


def state_action_processing(state, action = 0): 
	action_oh = np.asarray(sess.run(tf.one_hot([action], action_size)[0])).reshape(1,action_size)
	state = np.asarray(state).reshape(1, state_dim)
	return state, action_oh, [state, action_oh]

sess = tf.Session()
K.set_session(sess)

dims = [10,10]
start = [1,1]
goal = [3,3]
action_list = [0,1,2,3]

maze = env.simple_maze(dims, start, goal, action_list)

state_dim = len(maze.state_dim)
action_dim = 1
action_size = len(maze.action_space)

BATCH_SIZE = 3
TAU = 0.001     
LRA = 0.0001    
LRC = 0.001 
gamma = 0.9

# build the networks
actor = ActorNetwork(sess, state_dim, action_size, BATCH_SIZE, TAU, LRA)
critic = CriticNetwork(sess, state_dim, action_size, BATCH_SIZE, TAU, LRC)
returns = []

for episode in range(100): 

	R = 0
	state = maze.start
	maze.state = state
	print episode, state 

	for i in range(20):
		# get critic output and sample action
		state, _ , _ = state_action_processing(state)
		scores = actor.model.predict(state)[0]
		action =  np.where(np.random.multinomial(1,scores))[0][0]
		state, act, state_action = state_action_processing(state, action)
		
		# perform the action and observe state and reward
		next_state, reward = maze.take_action(action)
		#reward = maze.get_reward(action)
		#print str(i) + " Taking action " +  str(act) + " from state " + str(state) + " to state: " + str(next_state) + " with reward " + str(reward) 	

		# discounted return update 
		R += (gamma**i)*reward
	
		# get critic evaluation
		next_state, _ , _ = state_action_processing(next_state)
		next_action = np.where(np.random.multinomial(1,actor.model.predict(next_state)[0]))[0][0]
		_, act2, state_action2 = state_action_processing(next_state, next_action)
		v_next =  critic.model.predict(state_action2)[0]
		#print " SARSA action " +  str(next_action)

		if reward == 1:
			q_t = reward
		else:
			q_t = reward + gamma*v_next

		critic.model.fit(state_action, q_t, epochs=1, verbose=0) 
		grads = critic.gradients(state,act)
		actor.train(state, grads)

		if reward == 10: 
			break
		
		state = next_state

	print " "
	returns.append(R)

plt.plot(returns)
plt.show()
"""
#print sess.run(actor.state,feed_dict={actor.state: np.asarray([0,3,0]).reshape(1,3)})
state = [0,3,0]
actions = [0,0,1,0,0,0]
print "select action:"
scores = actor.model.predict(np.asarray(state).reshape(1, 3))[0]
print scores
action =  np.where(np.random.multinomial(1,scores))[0][0]
print action

print "critic: "

act = tf.one_hot([action], action_size)[0]
one_hot_action =  sess.run(act) 
#print one_hot_action
print  critic.model.predict([np.asarray(state).reshape(1, state_dim),np.asarray(one_hot_action).reshape(1,action_size)])[0]
grads = critic.gradients(np.asarray(state).reshape(1, state_dim),np.asarray(one_hot_action).reshape(1,action_size))
actor.train(np.asarray(state).reshape(1, state_dim), grads)
critic.model.fit([np.asarray(state).reshape(1, state_dim),np.asarray(one_hot_action).reshape(1,action_size)], np.asarray([1.2]))


if dones[k]:
	y_t[k] = rewards[k]
else:
       	y_t[k] = rewards[k] + GAMMA*target_q_values[k]


#print  critic.model.predict(np.asarray(state_action).reshape(1, 9))[0]
# load save files
#critic.model.save_weights("criticmodel.h5", overwrite=True)
#critic.model.load_weights("criticmodel.h5") #

# get action from state
#state = [0,3,0]
#scores =  actor.model.predict(np.asarray(state).reshape(1, 3))[0]
#action = np.where(np.random.multinomial(1,scores))[0][0]

# train
#a_for_grad = actor.model.predict(np.asarray(state).reshape(1, 3))[0]
#grads = critic.gradients(np.asarray(state).reshape(1,3), a_for_grad)
#print w
#print tf.gradients(a_for_grad, w)
"""	




