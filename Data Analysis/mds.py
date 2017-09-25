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

seed = np.random.RandomState(seed=3)


markers = ['o', 'v', 'h', 'H', 'o', 'v', 'h', 'H', 'h', 'H', 'o', 'v', 'h']
colors = ['b', 'r','g','c','y','m', 'b', 'r','g','c','y','m','y','m', 'b']
combs = list(itertools.product(markers, colors))



dirname = '../data'
users = os.listdir(dirname)
#D = np.zeros((len(pids), 24))
P = {}
D = []
user_models = []
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
		print user + '/' + session
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
	
		dd = -2*np.ones((4,3))
		for p in P:
			print P[p] 
			l = P[p].count(-1)
			w = P[p].count(1)
			if w == 0:
				v = 0.0
			else:
				if l > 0: 
					v = w/float(w+l)
				else: 
					v = 1.0

			dd[p[0]][p[1]] = v

		dd[dd == -2] = -1
		D.append(np.asarray(dd).flatten())
		
		um = []
		print scores
		for i in [1,2,3,4]: 
			if scores.count(i) == 0 and scores.count(-1*i) == 0: 
				um.append(-1.0)
			else: 
				um.append(scores.count(i)/float((scores.count(i) + scores.count(-1*i))))
		#print um
		user_models.append(um)

#print user_models
"""
D -= np.asarray(D).mean()
similarities = euclidean_distances(D)
mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed,dissimilarity="precomputed", n_jobs=1)
pos = mds.fit(similarities).embedding_
plt.subplot(121)
print pos
for i, p in enumerate(pos): 
	plt.plot(p[0], p[1], combs[i][0], markersize=9, color = combs[i][1], label = 'u'+str(i+1))
plt.title('MDS wrt difficulty and feedback')
#plt.legend()
#plt.show()
#plt.savefig('performance_RT_mds.jpg', bbox_inches='tight')		
#plt.close()
"""
U = user_models
user_models -= np.asarray(user_models).mean()
similarities = euclidean_distances(user_models)
mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed,dissimilarity="precomputed", n_jobs=1)
pos = mds.fit(similarities).embedding_
#plt.subplot(122)
for i, p in enumerate(pos): 
	plt.plot(p[0], p[1], combs[i][0], markersize=9, color = combs[i][1], label = 'u'+str(i+1))
plt.title('MDS wrt difficulty')
#plt.legend()
plt.show()
#plt.savefig('performance_RT_mds.jpg', bbox_inches='tight')		
plt.close()


X = np.asarray(U)

# CLUSTERING 
clusters = 3

kmeans = KMeans(n_clusters=clusters, random_state=0).fit(X)
print kmeans.labels_
print kmeans.cluster_centers_	
#u.writelines( "%s %s %s\n" % (item1, item2, item3) for item1, item2, item3 in zip(folders, user_models, kmeans.labels_) )
#m.writelines( "%s\n" % item for item in kmeans.cluster_centers_ )

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
plt.savefig('user_clusters.eps')
plt.close()

#with open('k' + str(clusters) + '/user_models.csv', 'w') as f:
#	writer = csv.writer(f,delimiter=',')
#	writer.writerows(user_models)

