#!/usr/bin/python
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.advanced_activations import PReLU
from keras.optimizers import RMSprop
import numpy as np 
from six.moves import cPickle

def build_pmodel(): 

	model = Sequential()
	model.add(Dense(10, input_dim = 5, activation='tanh'))
	model.add(Dense(10, activation='tanh'))
	model.add(Dense(1, activation='sigmoid'))
	rms = RMSprop()
	model.compile(loss='mse', optimizer='sgd')
	return model

def build_fmodel(): 

	model = Sequential()
	model.add(Dense(10, input_dim = 6, activation='linear'))
	model.add(Dense(10, activation='tanh'))
	model.add(Dense(1, activation='tanh'))
	rms = RMSprop()
	model.compile(loss='mse', optimizer='sgd')
	return model

"""
m = build_model()
s = (2,3,4,5)


y = np.zeros((1,5))
m.fit(np.asarray(s).reshape(1,4), y, batch_size=1, nb_epoch=100, verbose=0)
a = m.predict(np.asarray(s).reshape(1,4), batch_size = 1)[0]
print a

f = open('obj.save', 'wb')
cPickle.dump(m, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()

f = open('obj.save', 'rb')
loaded_obj = cPickle.load(f)
f.close()

a = loaded_obj.predict(np.asarray(s).reshape(1,4), batch_size = 1)[0]
print a
"""
