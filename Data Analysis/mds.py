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

U = user_models
user_models -= np.asarray(user_models).mean()
similarities = euclidean_distances(user_models)
mds = manifold.MDS(n_components=2, max_iter=300, eps=1e-3, random_state=seed, dissimilarity="precomputed", n_jobs=2)
pos = mds.fit(similarities).embedding_

for i, p in enumerate(pos): 
	plt.plot(p[0], p[1], combs[i][0], markersize=9, color = combs[i][1])
plt.title('MDS wrt difficulty')
plt.savefig("mds.png")
plt.close()

clusters = 3

# CLUSTERING on 2-D
kmeans = KMeans(n_clusters=clusters, random_state=0).fit(pos)
#kmeans = DBSCAN(eps = 0.1, min_samples=15).fit(pos)
print kmeans.labels_
mm = ['b', 'r', 'g', 'c']
for i, p in enumerate(pos): 
	plt.plot(p[0], p[1], 'o', markersize=9, color = mm[kmeans.labels_[i]])
	plt.text(p[0], p[1], n[i])
	#print name[i][0], kmeans.labels_[i]
plt.title('MDS - Kmeans')
plt.savefig('clustering.png')
plt.close()

# CLUSTERING on 4-D
CP = {}
for u, c in zip(U, kmeans.labels_):
	if CP.has_key(c): 
		CP[c].append(u)
	else: 
		CP[c] = [u]

cluster_means = {}
for cp in CP:
	cluster_means[cp] = np.asarray(CP[cp]).mean(axis = 0)
	 

#X = np.asarray(U)
#kmeans = KMeans(n_clusters=clusters, random_state=0).fit(X)
#print kmeans.labels_
#print kmeans.cluster_centers_	
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

plt.hold(False)
c = ['b', 'r', 'g', 'c']
for k in range(clusters):
	idx = index[k]
	bars = plt.bar(idx, cluster_means[k], 0.5, color=c[k], label = 'cluster_' + str(k+1))
	plt.hold(True)

if clusters == 1: 
	plt.xticks([0.75, 3.25, 5.75, 8.25], ('Level 1', 'L=5', 'L=7', 'L=9'))
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
plt.savefig('user_clusters.png')
plt.close()

f = open('datasets/user_clusters.csv', 'w')
for a,b in zip(name, kmeans.labels_):
	f.write(str(a[0]) + ' ' + str(b) + '\n')
f.close()

f = open('datasets/user_models.csv', 'w')
for a,b in zip(name, U):
	f.write(str(a[0]))
	for bb in b: 
		f.write(' ' + str(bb))
	f.write('\n')
f.close()

