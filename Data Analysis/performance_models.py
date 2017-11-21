#!/usr/bin/python
import numpy as np 
import os 
import re
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
import csv 
import sys
from sklearn.metrics import euclidean_distances
from sklearn import manifold
import random
import itertools
from sklearn.cluster import DBSCAN

seed = np.random.RandomState(seed=3)

markers = ['o', 'v', 'h', 'H', 'o', 'v', 'h', 'H', 'h', 'H', 'o', 'v', 'h']
colors = ['b', 'r','g','c','y','m', 'b', 'r','g','c','y','m','y','m', 'b']
combs = list(itertools.product(markers, colors))

fwrite = open('datasets/performance_models.csv', 'w')
fwrite.write("ID Level_1 Level_2 Level_3 Level_4 \n")


dirname = 'clean_data'
users = os.listdir(dirname)
P = {}
D = []
user_models = []
name = []
n = []
for user in users: 
	sessions = os.listdir(dirname + '/' + user)
	NoOfSessions = len(sessions)	
	for session in sessions:
		filename = dirname + '/' + user + '/' + session + '/logfile'
		f = open(filename, 'r')
		lines = f.readlines()
		f.close()
		P = {}
		scores = []
		for line in lines:
			a = re.split('\s+', line.strip())
			level = abs(int(a[3])) - 1
			perf = int(a[4])
			rf = int(a[2])
			key = tuple([level, rf])
			scores.append(int(a[3]))
			if P.has_key(key):
				P[key].append(perf)	
			else:
				P[key] = [perf]
		
		um = []
		for i in [1,2,3,4]: 
			if scores.count(i) == 0 and scores.count(-1*i) == 0: 
				um.append(-1.0)
			else: 
				um.append(scores.count(i)/float((scores.count(i) + scores.count(-1*i))))
		name.append([user + '/' + session])
		n.append([user])
		user_models.append(um)
	
		fwrite.write(user + '/' + session + ' ' + str(um[0]) + ' ' + str(um[1]) + ' ' + str(um[2]) + ' ' + str(um[3]) + '\n')
