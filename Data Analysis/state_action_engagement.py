#!/usr/bin/python
import os
import sys
import matplotlib.pyplot as plt
import numpy as np 
import re

D = [3,5,7,9]
dir1 = "clean_data/"

users = os.listdir(dir1)

for user in users:
	sessions = os.listdir(dir1 + user)
	for session in sessions: 
		print user, session
		filename = open(dir1 + user + '/' + session + '/state_action_engagement','w')
		logfile = open(dir1 + user + '/' + session + '/logfile','r')
		stateEEG = open(dir1 + user + '/' + session + '/state_normed_engagement', 'r')
		f1 = logfile.readlines()
		f2 = stateEEG.readlines()
		first = True
		for a,b in zip(f1,f2):
			A = re.split('\s+', a)[:-1]
			B = re.split('\s+', b)[:-1]

			state = B[0:3]
			length = state[0]
			rfeedback = state[1]
			pscore = state[2]
	
			pr_action = D.index(int(length))
			
			if int(rfeedback) == 1:
				pr_action = 4
			if int(rfeedback) == 2: 
				pr_action = 5
			
			if not first: 
				filename.write(str(pr_action) + '\n')
			else: 
				first = False

			engagement = np.asfarray(B[3::],float).mean()
			result = A[4]

			filename.write(str(length) + ' ' + str(rfeedback) + ' ' + str(pscore) + ' ' + str(engagement) + ' ')		

