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

from actor import ActorNetwork
from critic import CriticNetwork

sess = tf.Session()
K.set_session(sess)

state_dim = 3
action_dim = 1
action_size = 6
BATCH_SIZE = 3
TAU = 0.001     
LRA = 0.0001    
LRC = 0.001 

actor = ActorNetwork(sess, state_dim, action_size, BATCH_SIZE, TAU, LRA)
#print sess.run(actor.state,feed_dict={actor.state: np.asarray([0,3,0]).reshape(1,3)})
state = [0,3,0]
actions = [0,0,1,0,0,0]
print "select action:"
scores = actor.model.predict(np.asarray(state).reshape(1, 3))[0]
print scores
action =  np.where(np.random.multinomial(1,scores))[0][0]
print action 

print "critic: "
critic = CriticNetwork(sess, state_dim, action_size, BATCH_SIZE, TAU, LRC)
print  critic.model.predict([np.asarray(state).reshape(1, 3),np.asarray(actions).reshape(1, 6)])[0]

"""
if dones[k]:
	y_t[k] = rewards[k]
else:
       	y_t[k] = rewards[k] + GAMMA*target_q_values[k]
"""

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
	




