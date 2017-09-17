#!/usr/bin/python
import numpy as np
import math
#from keras.initializers import normal, identity
from keras.models import model_from_json, load_model
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, merge, Lambda, Activation
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
from keras.layers import concatenate

# estimates Q(s,a)
class CriticNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size
	
        K.set_session(sess)

        self.model, self.state, self.action, self.weights = self.create_critic_network(state_size, action_size)  
        #self.target_model, self.target_action, self.target_state = self.create_critic_network(state_size, action_size)  
        self.action_grads = tf.gradients(self.model.output, self.action)  #GRADIENTS for policy update
        self.sess.run(tf.global_variables_initializer())

    def gradients(self, state, action):
	#print state, action
        return self.sess.run(self.action_grads, feed_dict={self.state: state, self.action: action})[0]

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in xrange(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self, state_size, action_size):
        S = Input(shape=[state_size])  
	A = Input(shape=[action_size]) 
	S_A = concatenate([S,A])  
        h0 = Dense(5, activation='linear')(S_A)
        Q = Dense(1,activation='linear')(h0)   
        model = Model(inputs=[S,A],outputs=Q)
        #model = Model(inputs=S,outputs=Q)
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        return model, S, A, model.trainable_weights

