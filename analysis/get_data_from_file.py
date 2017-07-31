#!/usr/bin/python
import os
import sys
import matplotlib.pyplot as plt
import numpy as np 
import matplotlib.animation as animation
import re

def mean2d(v): 
	return [float(sum(l))/len(l) for l in zip(*v)]

def read_from_file(f):
	a, b, g, d, t, c = [], [], [], [], [], []
	lines = f.readlines()
	for line in lines:	
		w = line.split()
		if w[0] == 'a': 
			a.append([float(w[1]), float(w[2]), float(w[3]), float(w[4])])
		elif w[0] == 'b': 
			b.append([float(w[1]), float(w[2]), float(w[3]), float(w[4])])
		elif w[0] == 'g': 
			g.append([float(w[1]), float(w[2]), float(w[3]), float(w[4])])
		elif w[0] == 'd': 		
			d.append([float(w[1]), float(w[2]), float(w[3]), float(w[4])])
		elif w[0] == 't': 
			t.append([float(w[1]), float(w[2]), float(w[3]), float(w[4])])
		elif w[0] == 'c': 
			c.append(float(w[1]))
		
	
	return np.asarray(a).mean(axis=1), np.asarray(b).mean(axis=1),np.asarray(g).mean(axis=1),np.asarray(d).mean(axis=1),np.asarray(t).mean(axis=1), c
	#return mean2d(a),mean2d(b),mean2d(g),mean2d(d),mean2d(t),c
	#return a,b,g,d,t,c
	#float(sum(c)/len(c))

def ewma(Y, a = 0.2): 
	S = []
	#print Y
	for i, y in enumerate(Y): 
		if i == 0: 
			S.append(y)
		else: 
			S.append(a*Y[i-1] + (1-a)*S[i-1])
	return S	

dirname = "data/"
users = os.listdir(dirname)
print users


ENG = {} 
for user in users:
	
	sessions = os.listdir(dirname + '/' + user)
	for session in sessions:
		
		file_name = dirname + '/' + user + '/' + session
		logfile = open(file_name + '/state_EEG', 'r')
		lines = logfile.readlines()
		logfile.close()
		efile = open(file_name + '/engagement','w')
		for line in lines: 
			A = re.split('\s+', line)
			eeg_filename = A[3]
			print "opening: " + file_name + '/' + eeg_filename
			f = open(file_name + '/' + eeg_filename, 'r')
			a, b, g, d, t, c = read_from_file(f)
			c_smoothed = ewma(c)
			a_smoothed = ewma(a)
			b_smoothed = ewma(b)
			t_smoothed = ewma(t)

			e = [x+y for x, y in zip(a_smoothed, t_smoothed)]
			engagement = [x/y for x, y in zip(b_smoothed, e)]
			
			efile.write(str(A[0]) + ' ' + str(A[1]) + ' ' + str(A[2]))
			for E in engagement: 
				efile.write(' ' + str(E))

			if ENG.has_key(A[0]):
				ENG[A[0]].append(np.asarray(engagement).mean())	
			else:
				ENG[A[0]] = [np.asarray(engagement).mean()]

			efile.write('\n')
		#print ENG
		#raw_input()

A = []
W = []
L = []
for level in ENG: 
	a = ENG[level]
	weights = 100*np.ones_like(a)/len(a)
	A.append(a)
	W.append(weights)
	L.append('Level = ' + str(level))

	plt.hist(a, 2, weights = weights, label = 'Level = ' + str(level))
#plt.xlim([0,1])
	plt.legend()
	plt.show('L.png')
	plt.close()
	print "END"
		#raw_input()

				

			
		
		
