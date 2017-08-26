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
action_dim = 6
BATCH_SIZE = 3
TAU = 0.001     
LRA = 0.0001    
LRC = 0.001 

actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)

critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)

# load save files
#critic.model.save_weights("criticmodel.h5", overwrite=True)
#critic.model.load_weights("criticmodel.h5") #

# get action from state
state = [0,3,0]
#scores =  actor.model.predict(np.asarray(state).reshape(1, 3))[0]
#action = np.where(np.random.multinomial(1,scores))[0][0]

# train

	




