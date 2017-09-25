#!/usr/bin/python
import numpy as np 
import os 
import re
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
import csv 
import sys


root = '/home/tsiakas/Sequence_Learning/new/'
dirname = root + 'data'
users = os.listdir(dirname)


CLUSTERS = [1,2,3,4]
#clusters = 1
for clusters in CLUSTERS: 
	user_models = []
	peformances = []
	folders = []
	u = open('k' + str(clusters)+ '/user_clusters', 'w')
	m = open('k' + str(clusters)+ '/cluster_means', 'w')
	for user in users: 
		sessions = os.listdir(dirname + '/' + user)	
		for session in sessions: 
			scores = []
			filename = dirname + '/' + user + '/' + session + '/state_EEG'
			f = open(filename, 'r')
			lines = f.readlines()
			f.close()
			for line in lines:
				a = re.split('\s+', line.strip())
				scores.append(float(a[2]))
			um = []
			for i in [1.0,2.0,3.0,4.0]: 
				if scores.count(i) == 0 and scores.count(-1*i) == 0: 
					um.append(0.0)
				else: 
					um.append(scores.count(i)/float((scores.count(i) + scores.count(-1*i))))
			folders.append(user + '/' + session)
			user_models.append(um)

	X = np.asarray(user_models)


	# CLUSTERING 

	kmeans = KMeans(n_clusters=clusters, random_state=0).fit(X)
	print kmeans.labels_
	print kmeans.cluster_centers_	
	u.writelines( "%s %s %s\n" % (item1, item2, item3) for item1, item2, item3 in zip(folders, user_models, kmeans.labels_) )
	m.writelines( "%s\n" % item for item in kmeans.cluster_centers_ )

	if clusters == 1: 
		index = [[0.5, 3, 5.5 , 8], [1, 3.5, 6, 8.5], [1.5, 4, 6.5, 9], [2, 4.5, 7, 9.5]]
	if clusters == 2: 
		index = [[0.5, 3, 5.5 , 8], [1, 3.5, 6, 8.5], [1.5, 4, 6.5, 9], [2, 4.5, 7, 9.5]]
	if clusters == 3: 
		index = [[0.5, 3, 5.5 , 8], [1, 3.5, 6, 8.5], [1.5, 4, 6.5, 9], [2, 4.5, 7, 9.5]]
	if clusters == 4: 
		index = [[0.5, 3, 5.5 , 8], [1, 3.5, 6, 8.5], [1.5, 4, 6.5, 9], [2, 4.5, 7, 9.5]]


	c = ['r', 'b', 'g', 'y']
	for k in range(clusters):
		idx = index[k]
		bars = plt.bar(idx, np.asarray(kmeans.cluster_centers_[k]), 0.5, color=c[k], label = 'user_cluster_' + str(k+1))
		plt.hold(True)
	
	if clusters == 1: 
		plt.xticks([0.75, 3.25, 5.75, 8.25], ('L=3', 'L=5', 'L=7', 'L=9'))
		plt.xlim([0,9])
	if clusters == 2: 
		plt.xticks([1, 3.5, 6, 8.5], ('L=3', 'L=5', 'L=7', 'L=9'))
		plt.xlim([0,9.5])
	if clusters == 3: 
		plt.xticks([1.25, 3.75, 6.25, 8.75], ('L=3', 'L=5', 'L=7', 'L=9'))
		plt.xlim([0,9.75])
	if clusters == 4: 
		plt.xticks([1.5, 4, 6.5, 9], ('L=3', 'L=5', 'L=7', 'L=9'))
		plt.xlim([0,11])

	plt.ylim([0,1.2])

	plt.xlabel('Sequence Length')
	plt.ylabel('P(success|L)')
	plt.title('User Cluster Centroids (K = '+ str(clusters) +')')
	plt.legend()
	plt.tight_layout()
	plt.savefig('k' + str(clusters)+ '/user_clusters.eps')
	plt.close()

with open('k' + str(clusters) + '/user_models.csv', 'w') as f:
	writer = csv.writer(f,delimiter=',')
	writer.writerows(user_models)


