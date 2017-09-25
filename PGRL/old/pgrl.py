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

def moving_average(a, n=50) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def state_action_processing(state, action = 0): 
	action_oh = np.asarray(sess.run(tf.one_hot([action], action_size)[0])).reshape(1,action_size)
	state = 0.1*np.asarray(state).reshape(1, state_dim)
	return state, action_oh, [state, action_oh]


sess = tf.Session()
K.set_session(sess)

dims = [10,10]
start = [1,1]
goal = [3,3]
pits = [[1,2], [1,0], [3,5], [0,1], [1,3], [3,1], [4, 4], [5,0], [5,1], [5,2], [5,3]]
action_list = [0,1,2,3]

maze = env.simple_maze(dims, start, goal, pits, action_list)

state_dim = len(maze.state_dim)
action_dim = 1
action_size = len(maze.action_space)
print "action size " + str(action_size)
BATCH_SIZE = 3
TAU = 0.001     
LRA = 0.005    
LRC = 0.01
gamma = 0.9

# build the networks
actor = ActorNetwork(sess, state_dim, action_size, BATCH_SIZE, TAU, LRA)
critic = CriticNetwork(sess, state_dim, action_size, BATCH_SIZE, TAU, LRC)
returns = []
BATCH = np.empty((0,4))
maze.print_maze()

for episode in range(200): 
	R = 0
	state = maze.start
	maze.state = state

	for i in range(30):
		# get actor output and sample action
		st, _ , _ = state_action_processing(state)
		scores = actor.model.predict(st)[0]
		action =  np.where(np.random.multinomial(1,scores))[0][0]
		st, act, state_action = state_action_processing(state, action)

		# perform the action and observe state and reward
		next_state, reward, done = maze.take_action(action)
		#print state, action, reward 
		#print scores
		#maze.print_maze()
		# discounted return update 
		R += (gamma**i)*reward
	
		# get critic evaluation
		next_st, _ , _ = state_action_processing(next_state)
		next_action = np.where(np.random.multinomial(1,actor.model.predict(next_st)[0]))[0][0]
		_ , act2, state_action2 = state_action_processing(next_state, next_action)
		v_next =  critic.model.predict(state_action2)[0]

		# estimate Q(s,a)
		if done:
			v_target = reward
		else:
			v_target = reward + gamma*v_next

		# estimate TD-error
		td_error = v_target - critic.model.predict(state_action)[0]

		# check data type
		if type(v_target) is not np.ndarray: 
			v_target = np.asarray([v_target])

		# minibatch 
		BATCH = np.vstack( (BATCH, [st, act, state_action, v_target]) )
		if BATCH.shape[0] > 5: 
			batch = 
		# train critic model based on TD error
		critic.model.fit(state_action, v_target, epochs=1, verbose=0) 

		# estimate action gradients
		grads = critic.gradients(st, act)		
		
		# train actor network based on gradients
		actor.train(st, grads)
	
		# next transition
		state = next_state

		# reset episode
		if done: 
			#print state, action, actor.model.predict(st)[0]
			#if reward == 1: 
			#	print "SUCCESS " + str(critic.model.predict(state_action)[0]) + ' ' + str(v_target)
			#else: 
			#	print "FAILURE " + str(critic.model.predict(state_action)[0]) + ' ' + str(v_target)
			break

	print episode, i, R
	returns.append(R)
	

plt.plot(moving_average(returns))
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




