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

def ewma(Y, a = 0.05): 
	S = []
	for i, y in enumerate(Y): 
		if i == 0: 
			S.append(y)
		else: 
			S.append(a*Y[i-1] + (1-a)*S[i-1])
	return S	

clusters = load_clusters('datasets/user_clusters.csv')

if not os.path.exists('EEG_analysis/'):
	os.makedirs('EEG_analysis/')

D = [3,5,7,9]
indices = [1,2,3,4,5]
dirname = "clean_data"
users = os.listdir(dirname)
for index in indices: 
	efile = open('datasets/index_' + str(index) + '.csv','w')
	efile.write('ID cluster length robot_feedback previous_score current_result current_score engagement action\n')
	F = 1 
	FULL = []
	means = []
	for user in users:
		sessions = os.listdir(dirname + '/' + user)
		for session in sessions:
			EE = []
			turn_mean = []
			i = 0

			if not os.path.exists('EEG_analysis/' + user + '/' + session):
				os.makedirs('EEG_analysis/' + user + '/' + session)

			ff = open('EEG_analysis/' + user + '/' + session + '/index_' + str(index), 'w')
		
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
						#ff.write(' -1\n')
						last = 0
					else: 
						efile.write(' ' + str(action) + '\n')
						ff.write(' ' + str(action) + '\n')

				f = open(file_name + '/' + eeg_filename, 'r')
				a, b, g, d, t, c = read_from_file(f)
				a_smoothed = ewma(a)
				b_smoothed = ewma(b)
				t_smoothed = ewma(t)
				concentration = ewma(c)

				if index == 1: 
					# index 1
					e = [x+y for x, y in zip(a_smoothed, t_smoothed)]
					engagement = [x/y for x, y in zip(b_smoothed, e)]
				elif index == 2: 
					# index 2
					engagement = [x/y for x, y in zip(b_smoothed, t_smoothed)]
				elif index == 3: 
					# index 3
					engagement = [x/y for x, y in zip(b_smoothed, a_smoothed)]
				elif index == 4: 
					# index 4
					engagement = [x/y for x, y in zip(t_smoothed, a_smoothed)]
				elif index == 5: 
					# Muse built-in concentration
					engagement = concentration 

				i = i + len(engagement)
				turn_mean.append(np.asarray(engagement).mean())
				means.append(np.asarray(engagement).mean())			

				efile.write(user + '/' + session + ' ' + clusters[user + '/' + session]  + ' ' + str(length) + ' ' + str(rf) + ' ' + str(ps) + ' ' + str(result) + ' ' + str(score) + ' ' + str(np.asarray(engagement).mean()))
				ff.write(user + '/' + session + ' ' + clusters[user + '/' + session]  + ' ' + str(length) + ' ' + str(rf) + ' ' + str(ps) + ' ' + str(result) + ' ' + str(score))
			
				for E in engagement: 
					#efile.write(' ' + str(E))
					ff.write(' ' + str(E))
					FULL.append(E)
					EE.append(E)
				#plt.axvline(x=i-1, color = 'k')
				#plt.hold(True)		

			
			ff.write(' -1\n')
			ff.close() 	
			
			plt.plot(range(1,26), turn_mean)
			plt.title('Mean engagement per turn')
			plt.xlabel('Turns')
			plt.xlim([1,25])
			plt.savefig('EEG_analysis/' + user + '/' + session + '/index_' + str(index) + '.png')
			plt.hold(False)
			plt.close()
			

	weights = np.ones_like(means)/float(len(means))
	plt.hist(means, bins = 10, weights = weights)
	plt.title('engagement means - index ' + str(index))
	plt.savefig('EEG_analysis/index_' + str(index) + '.png')
