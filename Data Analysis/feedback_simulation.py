#!/usr/bin/python
import os
import sys
import matplotlib.pyplot as plt
import numpy as np 
import matplotlib.animation as animation
import re
import pandas as pd
from collections import Counter
from sklearn.svm import SVR
import itertools
from sklearn.metrics import mean_squared_error
from sklearn.svm import NuSVR
from sklearn.neural_network import MLPRegressor
import ann


CLF = 'asvr'

data = pd.read_csv('datasets/index_1.csv', delimiter=' ')				
C = data[['cluster','engagement', 'length','robot_feedback', 'previous_score', 'current_score' , 'current_result', 'action']]

# cluster data
C0 = C.loc[C['cluster']==0]
C1 = C.loc[C['cluster']==1]
C2 = C.loc[C['cluster']==2]

P0 = C0[['engagement', 'length','robot_feedback', 'previous_score', 'current_result']]
P1 = C1[['engagement', 'length','robot_feedback', 'previous_score', 'current_result']]
P2 = C2[['engagement', 'length','robot_feedback', 'previous_score', 'current_result']]

a0 =  P0.groupby(['length','robot_feedback', 'previous_score', 'current_result'])
a1 =  P1.groupby(['length','robot_feedback', 'previous_score', 'current_result'])
a2 =  P2.groupby(['length','robot_feedback', 'previous_score', 'current_result'])

D = [3,5,7,9]
S = [-4,-3,-2,-1,0,1,2,3,4]
L = [0.25, 0.5, 0.75, 1.0]
RF = [[1.0,0.0,0.0], [0.0,1.0,0.0], [0.0,0.0,1.0]]
PS = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
combs = (L, RF, PS)
states = list(itertools.product(*combs))

state_level = []
tmp = 0 
for i, s in enumerate(states):
	if tmp < s[0]:
		state_level.append(i)
		tmp = s[0]	  

for ii, cluster in enumerate([a0,a1,a2]):
	train_X = []
	train_Y = []
	for key, item in cluster:
		A = cluster.get_group(key)
		if A['engagement'].mean() < 1.0:
			training = [ L[D.index(key[0])], RF[key[1]][0],  RF[key[1]][1], RF[key[1]][2], PS[S.index(key[2])], key[3] ]
			target = A['engagement'].mean()
			train_X.append(training)
			train_Y.append(target)

	if CLF == 'svr': 
		clf = SVR(kernel='poly', C=50.0)
		clf.fit(train_X, train_Y)
		svr_prediction = clf.predict(train_X)
	    	svr_mse = mean_squared_error(train_Y, svr_prediction)
	    	svr_rmse1 = np.sqrt(svr_mse)
	    	print "\t\tTraining Data: SVR Root Mean Square Error = {0:0.2f}".format(svr_rmse1)
	else: 
		clf = ann.build_fmodel()
		x = np.asarray(train_X)
		y = np.asarray(train_Y)
		clf.fit(x, y, epochs=10000, verbose=0)

	print ii

	results = [-1.0, 1.0]
	labels = ["failure", "sucess"]
	outcome = ['or', 'oy']
	vmax = [] 
	for i, res in enumerate(results): 
		preds = []
		for s in states:
			if CLF == 'svr':
				preds.append(clf.predict(np.asarray([s[0], s[1][0], s[1][1], s[1][2], s[2], res]).reshape(1,6)))
			else: 
				preds.append(clf.predict(np.asarray([s[0], s[1][0], s[1][1], s[1][2], s[2], res]).reshape(1,6))[0][0])

		vmax.append(max(preds))
		
		plt.hold(True)
		plt.plot(preds, label = labels[i])
		
		#first = 1
		#for a,b in zip(train_X, train_Y): 
		#	if first: 
		#		plt.plot(states.index(tuple([a[0], [a[1], a[2], a[3]], a[4]])), b, outcome[i], label = 'real values')	
		#		first = 0
		#	else: 
		#		plt.plot(states.index(tuple([a[0], [a[1], a[2], a[3]], a[4]])), b, outcome[i])	
		#plt.show()
	
	height = max(vmax) + 0.1
	plt.bar(left=state_level, height=[height,height,height,height], width=27, color=['k','w', 'k', 'w'], alpha=0.1)
	plt.ylim([0, height])
	plt.xlim([0,len(preds) - 1])
	plt.legend()
	plt.savefig('feedback_c' + str(ii) + '.png')
	plt.close()


