#!/usr/bin/python
import os
import sys
import matplotlib.pyplot as plt
import numpy as np 
import matplotlib.animation as animation
import re
import operator

def mean2d(v): 
	return [float(sum(l))/len(l) for l in zip(*v)]

def read_from_file(f):
	a, b, g, d, t, c, h = [], [], [], [], [], [], []
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
		elif w[0] == 'h': 
			h.append([float(w[1]), float(w[2]), float(w[3]), float(w[4])])
		
	
	return np.asarray(a).mean(axis=1), np.asarray(b).mean(axis=1),np.asarray(g).mean(axis=1),np.asarray(d).mean(axis=1),np.asarray(t).mean(axis=1), c, h

def ewma(Y, a = 0.2): 
	S = []
	for i, y in enumerate(Y): 
		if i == 0: 
			S.append(y)
		else: 
			S.append(a*Y[i-1] + (1-a)*S[i-1])
	return S	


D = [3,5,7,9]
dirname = "clean_data/"
users = os.listdir(dirname)
full_means = 'logfiles/mean_engagement_per_level'
for user in users:
	sessions = os.listdir(dirname + '/' + user)
	for session in sessions:
		ts = 0 
		ENG = {}
		EE = []
		points = []
		annotation = []

		file_name = dirname + '/' + user + '/' + session
		logfile = open(file_name + '/state_EEG', 'r')
		lines = logfile.readlines()
		logfile.close()
		efile = open(file_name + '/state_engagement','w')
		T = []
		
		if not os.path.exists('Feedback Model/engagement/' + user + '/' + session):
   			os.makedirs('Feedback Model/engagement/'+ user + '/' + session)
		ff = open('Feedback Model/engagement/' + user + '/' + session + '/status', 'w')
		print "opening: " + file_name
		for line in lines: 
			A = re.split('\s+', line)
			eeg_filename = A[3]
			f = open(file_name + '/' + eeg_filename, 'r')
			a, b, g, d, t, c, h = read_from_file(f)
			ff.write(str(status) + '\n')
			
			c_smoothed = ewma(c)
			a_smoothed = ewma(a)
			b_smoothed = ewma(b)
			t_smoothed = ewma(t)

			e = [x+y for x, y in zip(a_smoothed, t_smoothed)]
			engagement = [x/y for x, y in zip(b_smoothed, e)]

			points.append(len(engagement)-1)
			efile.write(str(A[0]) + ' ' + str(A[1]) + ' ' + str(A[2]))
	
			for E in engagement: 
				efile.write(' ' + str(E))
				EE.append(E)
		
			annotation.append(A[0])
			if ENG.has_key(A[0]):
				ENG[A[0]] = np.append(ENG[A[0]], np.asarray(engagement))	
			else:
				ENG[A[0]] = np.asarray(engagement)

			efile.write('\n')

		ff.close()
		efile.close()
		
		if not os.path.exists('Feedback Model/engagement/' + user + '/' + session):
   			os.makedirs('Feedback Model/engagement/'+ user + '/' + session)

		plt.plot(EE)
		plt.hold(True)

		pr = 0 	
		for pp, ann in zip(points, annotation):
			plt.axvline(int(pp + pr), color = 'r')
			plt.text(int(pr) + 1, max(EE), ann)
			pr += pp 

			
		plt.savefig('Feedback Model/engagement/' + user + '/' + session + '/engagement.png')
		plt.hold(False)

		# plot normalized
		x = np.asarray(EE)
		minx = min(x)
		maxx = max(x)
		normed = (x-min(x))/(max(x)-min(x))
		plt.plot(normed)
		plt.ylim([-0.1, 1.1])
		plt.hold(True)

		pr = 0 
		for pp, ann in zip(points, annotation):
			plt.axvline(int(pp + pr), color = 'r')
			plt.text(int(pr) + 1, max(EE), ann)
			pr += pp
		plt.savefig('Feedback Model/engagement/' + user + '/' + session + '/engagement_normed.png')
		plt.hold(False)	

		flatten = lambda l: [item for sublist in l for item in sublist]
		Means = [0,0,0,0]
		Vars = [0,0,0,0]
		ff = open('Feedback Model/engagement/' + user + '/' + session + '/mean_engagement_per_level', 'w')
		for level in ENG: 
			a = ENG[level]
			normed_a = (a-minx)/(maxx-minx)
			weights = 100*np.ones_like(normed_a)/len(normed_a)
			plt.hist(normed_a, weights = weights, label = 'Level = ' + str(level))
			plt.legend()
			plt.title("M = " + str(np.asarray(normed_a).mean()) + ' var = ' + str(np.asarray(normed_a).var()))
			plt.savefig('Feedback Model/engagement/' + user + '/' + session + '/L_' + str(level) + '.png')
			Means[D.index(int(level))] = np.asarray(normed_a).mean()
			Vars[D.index(int(level))] = np.asarray(normed_a).var()

		plt.plot(Means)
		plt.savefig('Feedback Model/engagement/' + user + '/' + session + '/means.png')
		for m in Means: 
			ff.write(str(m) + '\n')
		ff.close()

		efile = open(file_name + '/state_engagement','r')
		lines = efile.readlines()
		efile.close()
		nfile = open(file_name + '/state_normed_engagement','w')

		for line in lines: 
			A = re.split('\s+', line)
			length = A[0]
			robot_feedback = A[1]
			previous_score = A[2]
			engagement = np.asfarray(A[3:-1],float)
			normed_engagement = (engagement-minx)/(maxx-minx)
			nfile.write(str(length) + ' ' + str(robot_feedback) + ' ' + str(previous_score))
			for n in normed_engagement: 
				nfile.write(' ' + str(n))
			nfile.write('\n')

