#!/usr/bin/python
import os
import sys
import matplotlib.pyplot as plt
import numpy as np 
import matplotlib.animation as animation
import re

def read_from_file(f):
	h =  []
	lines = f.readlines()
	for line in lines:	
		w = line.split()
		if w[0] == 'h': 
			h.append([float(w[1]), float(w[2]), float(w[3]), float(w[4])])
	return h

def check_status(V): 
	good = 0
	for v in V:
		if v == [1.0,1.0,1.0,1.0]:
			good +=1
	return good/float(len(V))	

dirname = "../data/"
users = os.listdir(dirname)

output = 'EEG/status'
ofile = open(output, 'w')

sess = 0

for user in users:
	sessions = os.listdir(dirname + '/' + user)
	for session in sessions:
		sess += 1
		S = []
		file_name = dirname + '/' + user + '/' + session
		logfile = open(file_name + '/state_EEG', 'r')
		lines = logfile.readlines()
		logfile.close()
		
		for line in lines: 
			A = re.split('\s+', line)
			eeg_filename = A[3]
			print "opening: " + file_name + '/' + eeg_filename
			f = open(file_name + '/' + eeg_filename, 'r')
			h = read_from_file(f)
			s = check_status(h)
			S.append(s)
			ofile.write(user + ' ' + session + ' ' + eeg_filename + ' ' + str(s) + '\n')
	
		ofile.write(str(sum(S)/float(len(S))) + '\n\n')

print sess			
	
			
