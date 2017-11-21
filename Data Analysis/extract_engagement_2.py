#!/usr/bin/python
import os
import sys
import matplotlib.pyplot as plt
import numpy as np 
import matplotlib.animation as animation
import re
from collections import Counter

def normalize_by_range(x, nmin = 0, nmax = 1):
	x = np.asarray(x)
	return (nmax-nmin)*(x-min(x))/(max(x)-min(x)) + nmin

def statistics(x):
	x = np.asarray(x)
	return min(x), max(x), x.mean()

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

def ewma(Y, a = 0.02): 
	S = []
	for i, y in enumerate(Y): 
		if i == 0: 
			S.append(y)
		else: 
			S.append(a*Y[i-1] + (1-a)*S[i-1])
	return S	

#clusters = load_clusters('datasets/user_clusters.csv')

if not os.path.exists('EEG_analysis/'):
	os.makedirs('EEG_analysis/')

D = [3,5,7,9]
indices = [1,2,3,4,5]
indices = [1]
dirname = "clean_data"
users = os.listdir(dirname)


for index in indices: 
	print "index " + str(index)
		
	file1 = open('datasets/index_' + str(index) + '.csv','w')
	file1.write('ID length robot_feedback previous_score current_result current_score engagement action\n')
	
	file2 = open('EEG_analysis/index_' + str(index) + '.csv', 'w')

	F = 1 
	FULL = []
	means = []
	for user in users:
		sessions = os.listdir(dirname + '/' + user)
		for session in sessions:
			EE = []
			turn_mean = []
			i = 0

			#if not os.path.exists('EEG_analysis/' + user + '/' + session):
			#	os.makedirs('EEG_analysis/' + user + '/' + session)
		
			file_name = dirname + '/' + user + '/' + session
			log1 = open(file_name + '/state_EEG', 'r')
			lines1 = log1.readlines()
			log1.close()

			log2 = open(file_name + '/logfile','r')
			lines2 = log2.readlines()
			log2.close()

			last = 1
			turn = [0]
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
						file1.write(' -1\n')
						last = 0
					else: 
						file1.write(' ' + str(action) + '\n')

				f = open(file_name + '/' + eeg_filename, 'r')
				a, b, g, d, t, c = read_from_file(f)
				a_smoothed = ewma(a)
				b_smoothed = ewma(b)
				t_smoothed = ewma(t)
				concentration = ewma(c)

				if index == 1: 
					# index 1 - b/a+t
					e = [x+y for x, y in zip(a_smoothed, t_smoothed)]
					engagement = [x/y for x, y in zip(b_smoothed, e)]
				elif index == 2: 
					# index 2 - b/t
					engagement = [x/y for x, y in zip(b_smoothed, t_smoothed)]
				elif index == 3: 
					# index 3 - b/a
					engagement = [x/y for x, y in zip(b_smoothed, a_smoothed)]
				elif index == 4: 
					# index 4 - t/a
					engagement = [x/y for x, y in zip(t_smoothed, a_smoothed)]
				elif index == 5: 
					# Muse built-in concentration
					engagement = concentration
				#elif index == 6: 
				# TODO add frontal assymentry	 
			
				#clength = len(engagement)
				i = i + len(engagement)
				turn.append(i)
				#turn_mean.append(np.asarray(engagement).mean())
				#means.append(np.asarray(engagement).mean())			

				file1.write(user + '/' + session + ' ' + str(length) + ' ' + str(rf) + ' ' + str(ps) + ' ' + str(result) + ' ' + str(score) + ' ' + str(np.asarray(engagement).mean()))
				file2.write(user + '/' + session + ' ' + str(length) + ' ' + str(rf) + ' ' + str(ps) + ' ' + str(result) + ' ' + str(score))

				for E in engagement: 
					file2.write(' ' + str(E))
					#FULL.append(E)
					EE.append(E)	
			
			#file2.write(' -1 \n')

		normed = normalize_by_range(EE)
		#plt.subplot(211)
		#plt.plot(EE)
		#plt.subplot(212)
		#plt.plot(normed)
		#plt.savefig('EEG_analysis/' + user + '/' + session + '/normed_' + str(index) + '.png')
		#plt.hold(False)
		#print turn
		#print len(normed)
		#raw_input()
	
		#aa = 1 
		#for ii, n in enumerate(normed):
		#	if ii < len(normed) and aa < len(turn): 
		#		#print ii, len(turn), aa
		#		if ii == turn[aa]: 
		#		#	print turn[aa]	
		#			file2.write('\n')
		#			aa += 1
		#	file2.write(str(n) + ' ')
		#file2.write('\n')
		#file2.close()

		# normalized index file
		#f1 = open('EEG_analysis/' + user + '/' + session + '/index_' + str(index), 'r')
		#f2 = open('EEG_analysis/' + user + '/' + session + '/normalized', 'r')
		
		#lines1 = f1.readlines()
		#lines2 = f2.readlines()
		#for a,b in zip(lines1, lines2):
		#	aa = a.split()
		#	bb = b.split()
		#	eng = np.asarray(bb).astype(float).mean()
		#	f3.write(str(aa[0])+ ' '+str(aa[1])+' '+str(aa[2])+' '+str(aa[3])+' '+str(aa[4])+' '+str(aa[5])+' '+str(aa[6])+' '+str(eng)+' '+str(aa[-1]) + '\n')
		#f3.close()

		#plt.plot(range(1,26), turn_mean)
		#plt.title('Mean engagement per turn')
		#plt.xlabel('Turns')
		#plt.xlim([1,25])
		#plt.savefig('EEG_analysis/' + user + '/' + session + '/index_' + str(index) + '.png')
		#plt.hold(False)
		#plt.close()
			

	#weights = np.ones_like(means)/float(len(means))
	#plt.hist(means, bins = 10, weights = weights)
	#plt.title('engagement means - index ' + str(index))
	#plt.savefig('EEG_analysis/index_' + str(index) + '.png')

