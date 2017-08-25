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
#critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
print actor

	




