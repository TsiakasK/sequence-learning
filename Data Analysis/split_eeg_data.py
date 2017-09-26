#!/usr/bin/python
import os
import sys
import matplotlib.pyplot as plt
import numpy as np 
import re

def mean2d(v): 
	return [float(sum(l))/len(l) for l in zip(*v)]

def read_from_file(fin, fout1, fout2):
	a, b, g, d, t, c = [], [], [], [], [], []
	lines = fin.readlines()
	for line in lines:	
		w = line.split()
		if w[0] == 'eeg': 
			#print [float(w[1]), float(w[2]), float(w[3]), float(w[4])]
			fout1.write(str(w[1]) + ' ' + str(w[2]) + ' ' + str(w[3]) + ' ' + str(w[4]) + '\n')
			
		else:
			if w[0] == 'c': 
				fout2.write(str(w[0]) + ' ' + str(w[1]) + ' 0 0 0\n')
			else: 
				fout2.write(str(w[0]) + ' ' + str(w[1]) + ' ' + str(w[2]) + ' ' + str(w[3]) + ' ' + str(w[4]) + '\n')
		"""
		elif w[0] == 'g': 
			g.append([float(w[1]), float(w[2]), float(w[3]), float(w[4])])
		elif w[0] == 'd': 		
			d.append([float(w[1]), float(w[2]), float(w[3]), float(w[4])])
		elif w[0] == 't': 
			t.append([float(w[1]), float(w[2]), float(w[3]), float(w[4])])
		elif w[0] == 'c': 
			c.append(float(w[1]))
		"""
	
	#return np.asarray(a).mean(axis=1), np.asarray(b).mean(axis=1),np.asarray(g).mean(axis=1),np.asarray(d).mean(axis=1),np.asarray(t).mean(axis=1), c
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

dirname = "clean_data/"
users = os.listdir(dirname)
print users

for user in users:
	
	sessions = os.listdir(dirname + '/' + user)
	for session in sessions:
		
		file_name = dirname + '/' + user + '/' + session
		if not os.path.exists(file_name + '/EEG'):
			os.makedirs(file_name + '/EEG') 
	
		logfile = open(file_name + '/state_EEG', 'r')
		lines = logfile.readlines()
		logfile.close()
		
		for line in lines: 
			A = re.split('\s+', line)
			eeg_filename_1 = A[3]
			eeg_filename_2 = A[4]
			print "opening: " + file_name + '/' + eeg_filename_1
			print "writing: " + file_name + '/EEG/' + eeg_filename_1
			f1 = open(file_name + '/' + eeg_filename_1, 'r')
			f2 = open(file_name + '/EEG/eeg_' + eeg_filename_1, 'w')
			f3 = open(file_name + '/EEG/bands_' + eeg_filename_1, 'w')
			read_from_file(f1,f2, f3)

			print "opening: " + file_name + '/' + eeg_filename_2
			print "writing: " + file_name + '/EEG/' + eeg_filename_2
			f1 = open(file_name + '/' + eeg_filename_2, 'r')
			f2 = open(file_name + '/EEG/eeg_' + eeg_filename_2, 'w')
			f3 = open(file_name + '/EEG/bands_' + eeg_filename_2, 'w')
			read_from_file(f1,f2, f3)
			


