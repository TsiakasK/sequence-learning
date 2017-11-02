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
import pandas as pd
from sklearn.cluster import DBSCAN
seed = np.random.RandomState(seed=3)

markers = ['o', 'v', 'h', 'H', 'o', 'v', 'h', 'H', 'h', 'H', 'o', 'v', 'h']
colors = ['b', 'r','g','c','y','m', 'b', 'r','g','c','y','m','y','m', 'b']
combs = list(itertools.product(markers, colors))
data = pd.read_csv('datasets/all_indices.csv', delimiter=',')
C = data[['ID','index1','length','current_result']]
users = C['ID'].unique()
clusters = 3

user_models = []
perf_models = []
eng_models = []
for user in users: 
	D = C.loc[C['ID']==user]
	um = [-1.0,-1.0,-1.0,-1.0, 0.0, 0.0, 0.0, 0.0]
	perf = [-1.0,-1.0,-1.0,-1.0]
	eng = [0.0, 0.0, 0.0, 0.0]	
	for i, l in enumerate([3,5,7,9]):
		L = D.loc[D['length']==l]
		wins = len(L.loc[L['current_result']==1]) 
		losses =  len(L.loc[L['current_result']==-1])
		if wins == 0: 
			um[i] = 0.0
			perf[i] = 0.0
		elif losses == 0:
			um[i] = 1.0
			perf[i] = 1.0
		else:
			um[i] = wins/float(wins+losses)
			perf[i] = wins/float(wins+losses)
		
		um[i+4] = L['index1'].mean()
		eng[i] = L['index1'].mean()
		
	user_models.append(um)
	perf_models.append(perf)
	eng_models.append(eng)	


labels = ["performance_engagement", "performance_based", "engagement_based"]
models = [user_models, perf_models, eng_models]	

for model, label in zip(models, labels): 
	model -= np.asarray(model).mean()
	similarities = euclidean_distances(model)
	mds = manifold.MDS(n_components=2, max_iter=300, eps=1e-6, random_state=seed, dissimilarity="precomputed", n_jobs=2)
	pos = mds.fit(similarities).embedding_

	for i, p in enumerate(pos): 
		plt.plot(p[0], p[1], combs[i][0], markersize=9, color = combs[i][1])
	plt.title('Multidimensional Scaling')
	plt.savefig(label + "_mds.png")
	plt.close()

	# CLUSTERING on 2-D
	kmeans = KMeans(n_clusters=clusters, random_state=0).fit(pos)
	#kmeans = DBSCAN(eps = 0.1, min_samples=15).fit(pos)
	print kmeans.labels_
	mm = ['b', 'r', 'g', 'c']
	first = [1,1,1]
	for i, p in enumerate(pos): 
		if first[kmeans.labels_[i]]: 
			plt.plot(p[0], p[1], 'o', markersize=9, color = mm[kmeans.labels_[i]], label = 'cluster_' + str(kmeans.labels_[i] + 1))
			first[kmeans.labels_[i]] = 0 
		else: 
			plt.plot(p[0], p[1], 'o', markersize=9, color = mm[kmeans.labels_[i]])

		#plt.text(p[0], p[1], n[i])
		#print name[i][0], kmeans.labels_[i]
	plt.legend()
	plt.title('Clustering using MDS')
	plt.savefig(label + '_clustering.png')
	plt.close()

