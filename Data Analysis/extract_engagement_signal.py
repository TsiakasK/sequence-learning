#!/usr/bin/python
import os
import sys
import matplotlib.pyplot as plt
import numpy as np 
import matplotlib.animation as animation
import re
from collections import Counter

# dict for user-clusterID
def load_clusters(filename):
	filehandler = open(filename, 'r')
	lines = filehandler.readlines()
	filehandler.close()
	Clusters = {}
	for line in lines:
		A = re.split('\s+', line)
		Clusters[A[0]] = A[1]

	return Clusters		

def read_from_file(f):
	a, b, g, d, t = [], [], [], [], []
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
	
	return np.asarray(a).mean(axis=1), np.asarray(b).mean(axis=1),np.asarray(g).mean(axis=1),np.asarray(d).mean(axis=1),np.asarray(t).mean(axis=1)

def ewma(Y, a = 0.1): 
	S = []
	for i, y in enumerate(Y): 
		if i == 0: 
			S.append(y)
		else: 
			S.append(a*Y[i-1] + (1-a)*S[i-1])
	return S	

clusters = load_clusters('datasets/user_clusters.csv')

dirname = "clean_data"
users = os.listdir(dirname)
efile = open('datasets/engagement','w')
D = [3,5,7,9]
F = 1 
for user in users:
	sessions = os.listdir(dirname + '/' + user)
	for session in sessions:
		EE = []

		file_name = dirname + '/' + user + '/' + session
		log1 = open(file_name + '/state_EEG', 'r')
		lines1 = log1.readlines()
		log1.close()

		log2= open(file_name + '/logfile','r')
		lines2 = log2.readlines()
		log2.close()

		last = 1
		for a,b in zip(lines1, lines2): 
			A = re.split('\s+', a)[:-1]
			B = re.split('\s+', b)[:-1]

			eeg_filename = A[3]
			length = A[0]
			rf = A[1]
			ps = A[2]
			result = B[4]
			score = int(B[4])*int(D.index(int(length)) + 1)

			action = D.index(int(length))
			if int(rf) == 1: 
				action = 4
			if int(rf) == 2: 
				action = 5

			if F:
				F = 0
				last = 0 
			else:  
				if last: 
					efile.write(' -1\n')
					last = 0
				else: 
					efile.write(' ' + str(action) + '\n')
	
			#print "opening: " + file_name + '/' + eeg_filename
			f = open(file_name + '/' + eeg_filename, 'r')
			a, b, g, d, t = read_from_file(f)
			a_smoothed = ewma(a)
			b_smoothed = ewma(b)
			t_smoothed = ewma(t)

			e = [x+y for x, y in zip(a_smoothed, t_smoothed)]
			engagement = [x/y for x, y in zip(b_smoothed, e)]
			
			efile.write(user + '/' + session + ' ' + clusters[user + '/' + session]  + ' ' + str(length) + ' ' + str(rf) + ' ' + str(ps) + ' ' + str(result) + ' ' + str(score))
	
			for E in engagement: 
				efile.write(' ' + str(E))

		
