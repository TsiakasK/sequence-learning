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
import csv 
import ann
import sys

def normalize(c): 
	features = np.asarray(c)
	normed_features = []
	for column in features.T:
		normalized = (column-min(column))/float(max(column)-min(column))
		normed_features.append(list(normalized))
	return np.asarray(normed_features).T

difficulty = [3,5,7,9]
feedback = [0,1,2]
score = [-4,-3,-2,-1,0,1,2,3,4]
combs = (difficulty, feedback, score)
states = list(itertools.product(*combs))

NoClusters = int(sys.argv[1])
clusters = []
cluster_data = []
root = '/home/tsiakas/Sequence_Learning/new/'
data_folder = root + 'data'
cluster_folder = 'k' + str(NoClusters)
cluster_file = root + 'user_models/' + cluster_folder + '/user_clusters'
f = open(cluster_file)
filedata = f.readlines()
f.close()

USER = [[] for x in range(NoClusters)]

for line in filedata:
	a = re.split('\s+', line.strip()) 
	USER[int(a[-1])].append(a[0])
	
for k in range(NoClusters): 
	T = [-1 * 1 for x in range(len(states))]
	state_score = {}
	state = {}
	users = USER[k]
	for user in users: 
		filename = data_folder + '/' + user +  '/logfile'
		statefile = data_folder + '/' + user + '/state_EEG'
		log = open(filename, 'r')
		lines = log.readlines()
		log.close()
		log2 = open(statefile, 'r')
		lines2 = log2.readlines()
		log2.close()

		for line, line2 in zip(lines, lines2): 
			a1 = re.split('\s+', line.strip())
			score = -1 if int(int(a1[-1]) < 0) else 1
			a2 = re.split('\s+', line2.strip())
			length = int(a2[0])
			rfeedabck = int(a2[1])
			previous_score = int(a2[2])
			bigram = str([length, rfeedabck, previous_score, score])
			unigram = str([length, rfeedabck, previous_score])

			if state_score.has_key(bigram):
				state_score[bigram] += 1
			else:
	    			state_score[bigram] = 1

			if state.has_key(unigram):
				state[unigram] += 1
			else:
	    			state[unigram] = 1

	#print state_score
	features = []
	for s in state:
		s = eval(s) 		
		win = [s[0], s[1], s[2], 1]
		lose = [s[0], s[1], s[2], -1]
		if state_score.has_key(str(win)):
			N1 = state_score[str(win)]
			N2 = state[str(s)]
			T[states.index(tuple(s))] = float(N1)/float(N2)
			a = [int(s[0]), int(s[1]), int(s[2]), T[states.index(tuple(s))]]
			features.append(a)
		elif state_score.has_key(str(lose)):
			N1 = state_score[str(lose)]
			N2 = state[str(s)]
			T[states.index(tuple(s))] = 1.0 - (float(N1)/float(N2))
			a = [int(s[0]), int(s[1]), int(s[2]), T[states.index(tuple(s))]]
			features.append(a)
	

	plt.hist(T)
	plt.savefig(cluster_folder + '/Phist_cluster_' + str(k))
	plt.close()

	probs = open(cluster_folder + '/P_success_given_state_' + str(k), 'w')
	train_X = []
	train_Y = []
	for line in features: 
		probs.write(str(line[0]) + ' ' + str(line[1]) + ' ' + str(line[2]) + ' ' + str(line[3]) + '\n')
		training = [line[0], line[1], line[2]]
		target = line[3]
		train_X.append(training)
		train_Y.append(target)
	probs.close()

	N = ann.build_model()
	x = normalize(train_X)
	y = np.asarray(train_Y)
	N.fit(x, y, epochs=10000, verbose=1)
	f = open(cluster_folder + '/performance_' + str(k) + '.model', 'wb')
	cPickle.dump(N, f, protocol=cPickle.HIGHEST_PROTOCOL)
	f.close()


