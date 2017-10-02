#!/usr/bin/python
import os
import sys
import matplotlib.pyplot as plt
import numpy as np 
import matplotlib.animation as animation
import re
import pandas as pd
from collections import Counter

FILE = 1

if FILE == 1: 
	data = pd.read_csv('../datasets/engagement_means.csv', delimiter=' ')				#1
if FILE == 2: 
	data = pd.read_csv('../datasets/engagement_means_normed_range.csv', delimiter=' ')		#2
if FILE == 3:
	data = pd.read_csv('../datasets/engagement_means_normed_mean.csv', delimiter=' ')		#3
if FILE == 4: 
	data = pd.read_csv('../datasets/engagement_means_normed_mean_range.csv', delimiter=' ')		#4


C0 = data.loc[data['cluster']==0]
C1 = data.loc[data['cluster']==1]
C2 = data.loc[data['cluster']==2]

C = [C0, C1, C2]


filename = open('c0_diff_' + str(FILE) + '.csv', 'w')
C0_users = C0['ID'].unique()
for u in C0_users: 
	first = 1
	tmp = C0.loc[C0['ID']==u][["length", "robot_feedback", "previous_score", "current_result", "action", "engagement"]]
	tmp = np.asarray(tmp) 
	for line in tmp: 
		#print line
		if first: 
			previous_engagement = float(line[5])
			previous_state_action = [int(line[0]), int(line[1]), int(line[2]), int(line[3]), int(line[4])]
			first = 0
			#print previous_engagement
		else: 
			filename.write(u + ' ')
			diff = float(line[5]) - previous_engagement
			previous_engagement = float(line[5])
			previous_state_action = [int(line[0]), int(line[1]), int(line[2]), int(line[3]), int(line[4])]
			for psa in previous_state_action: 
				filename.write(str(psa) + ' ')
			filename.write(str(diff) + '\n')
filename.close()

filename = open('c1_diff_' + str(FILE) + '.csv', 'w')
C1_users = C1['ID'].unique()
for u in C1_users: 
	first = 1
	tmp = C1.loc[C1['ID']==u][["length", "robot_feedback", "previous_score", "current_result", "action", "engagement"]]
	tmp = np.asarray(tmp) 
	for line in tmp: 
		#print line
		if first: 
			previous_engagement = float(line[5])
			previous_state_action = [int(line[0]), int(line[1]), int(line[2]), int(line[3]), int(line[4])]
			first = 0
			#print previous_engagement
		else: 
			filename.write(u + ' ')
			diff = float(line[5]) - previous_engagement
			previous_engagement = float(line[5])
			previous_state_action = [int(line[0]), int(line[1]), int(line[2]), int(line[3]), int(line[4])]
			for psa in previous_state_action: 
				filename.write(str(psa) + ' ')
			filename.write(str(diff) + '\n')
filename.close()

filename = open('c2_diff_' + str(FILE) + '.csv', 'w')
C2_users = C2['ID'].unique()
for u in C2_users: 
	first = 1
	tmp = C2.loc[C2['ID']==u][["length", "robot_feedback", "previous_score", "current_result", "action", "engagement"]]
	tmp = np.asarray(tmp) 
	print u
	for line in tmp: 
		#print line
		if first: 
			previous_engagement = float(line[5])
			previous_state_action = [int(line[0]), int(line[1]), int(line[2]), int(line[3]), int(line[4])]
			first = 0
			#print previous_engagement
		else: 
			filename.write(u + ' ')
			diff = float(line[5]) - previous_engagement
			print previous_state_action, diff
			previous_engagement = float(line[5])
			previous_state_action = [int(line[0]), int(line[1]), int(line[2]), int(line[3]), int(line[4])]
			for psa in previous_state_action: 
				filename.write(str(psa) + ' ')
			filename.write(str(diff) + '\n')
filename.close()

	        

