#!/usr/bin/python
import numpy as np 
import os 
import re
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from six.moves import cPickle
import itertools
import ann
import sys

def normalize(c): 
	features = np.asarray(c)
	normed_features = []
	for column in features.T:
		normalized = (column-min(column))/(max(column)-min(column))
		normed_features.append(list(normalized))
	return np.asarray(normed_features).T

difficulty = [3,5,7,9]
feedback = [0,1,2]
score = [-4,-3,-2,-1,0,1,2,3,4]
combs = (difficulty, feedback, score)
states = list(itertools.product(*combs))

statess =  [[float(y) for y in x] for x in states]
normed_states = normalize(statess)

NoClusters = int(sys.argv[1])
cluster_folder = 'k' + str(NoClusters)

for k in range(NoClusters): 
	T = [-1 * 1 for x in range(len(states))]

	f = open(cluster_folder + '/performance_' + str(k) + '.model', 'rb')
	N = cPickle.load(f)
	f.close()

	p = open(cluster_folder + '/P_success_given_state_' + str(k), 'r')
	lines = p.readlines()
	p.close()

	for line in lines:
		a = re.split('\s+', line.strip())
		state = [int(a[0]), int(a[1]), int(a[2])]
		T[states.index(tuple(state))] = float(a[3])

	a = []
	b = []
	for i, state in enumerate(states):  
		b.append(T[i])
		a.append(N.predict(np.asarray(normed_states[i,:]).reshape(1,3))[0][0])

	plt.plot(a, label = 'estimated')
	l = 0 	
	for i, point in enumerate(T): 
		plt.hold(True)
		if point > -1:
			if l == 0: 
				plt.plot(i, point, 'or', label = 'target') 
				l = 1
			plt.plot(i, point,  'or')

	plt.xlabel("states")
	plt.ylabel("P(sucess|state)")
	plt.legend()
	plt.savefig(cluster_folder + '/performance_network_' + str(k) + '.png')
	plt.close()

