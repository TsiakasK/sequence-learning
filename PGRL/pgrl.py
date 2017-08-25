#!/usr/bin/python
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.advanced_activations import PReLU
from keras.optimizers import RMSprop
import numpy as np 


class PolicyGradient(): 
	def __init__(self, state_input, actions, hidden, lr = 1e-03):  
		self.model = Sequential()
		self.state_input = state_input
		self.actions = actions
		self.hidden = hidden
		self.lr = lr 
		
		self.pmodel = PolicyModel(self.state_input,self.actions,self.hidden)
		self.vmodel = ValueModel(self.state_input,self.actions,self.hidden)


class PolicyModel():
	def __init__(self, state_input, actions, hidden, lr = 1e-03):  
		self.model = Sequential()
		self.state_input = state_input
		self.actions = actions
		self.hidden = hidden
		self.lr = lr 

		self.model.add(Dense(5, input_dim = self.state_input, activation='linear'))

		for x in range(self.hidden):
			self.model.add(Dense(5, activation='linear'))
	
		self.model.add(Dense(self.actions, activation='tanh'))
		self.model.compile(loss='mse', optimizer='sgd')

class ValueModel(): 
	def __init__(self, state_input, actions, hidden, lr = 1e-03):  
		self.model = Sequential()
		self.state_input = state_input
		self.actions = actions
		self.hidden = hidden

		self.model.add(Dense(5, input_dim = self.state_input + self.actions, activation='linear'))

		for x in range(self.hidden-1):
			self.model.add(Dense(5, activation='linear'))
	
		self.model.add(Dense(1, activation='tanh'))
		self.model.compile(loss='mse', optimizer='sgd')


pg = PolicyGradient(3,6,1)
print pg.vmodel.model.get_weights()
print pg.pmodel.model.get_weights()

