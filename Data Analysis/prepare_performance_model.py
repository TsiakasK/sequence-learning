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
import pickle
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
def display_scores(model, scores):
    print '\t---------- {} ----------'.format(model)
    print "\t\tScores: {}".format(scores)
    print "\t\tMean: {}".format(scores.mean())
    print "\t\tSTD: {}".format(scores.std())



dirname = '../../sequence-learning-master/Data Analysis/'
users = os.listdir(dirname)
data_files_per_cluster = {}
performance_models_per_cluster = []

# Read the clusters

clusters_file = dirname + 'clusters'
c_file = open(clusters_file, 'r')
lines = c_file.readlines()
c_file.close()

# Find files belonging to each cluster
for line in lines:
    f_name, cluster_no = re.split('\s+', line.strip())
    cluster_no = int(cluster_no)
    if data_files_per_cluster.has_key(cluster_no):
        data_files_per_cluster[cluster_no].append(f_name)
    else:
        data_files_per_cluster[cluster_no] = [f_name]

print 'There are {0} clusters'.format(len(data_files_per_cluster))

#Create Dictionary for the Performance Data for each Cluster
for cluster_i in data_files_per_cluster:

    attributes = {}

    for user_i in data_files_per_cluster[cluster_i]:

        user_i_file = dirname + 'clean_data/'+ user_i + '/logfile'
        f = open(user_i_file, 'r')
        lines = f.readlines()
        f.close()

        prev_score = 0
        for line in lines:
            a = re.split('\s+', line.strip())
            perf = int(a[4])  # 1 -1
            level = abs(int(a[3]))  # 1 2 3 4
            current_score = int(a[3])  # 1 -1 2 -2 3 -3 4 -4
            rb_feedback = int(a[2])
            key = tuple([level, rb_feedback, prev_score])
            prev_score = current_score

            if attributes.has_key(key):
                attributes[key].append(perf)
            else:
                attributes[key] = [perf]

    for row in attributes:
        loss= float(attributes[row].count(-1))
        win = float(attributes[row].count(1))
        success_rate = win / (win+loss)
        attributes[row] = success_rate

    performance_models_per_cluster.append(attributes)

with open('performance_models_per_cluster','w') as file:
    pickle.dump(performance_models_per_cluster,file)

print "Successfully saved Performance Models for {} Clusters!".format(len(performance_models_per_cluster))


""" Regression Model using NN """
#Perform NN on each cluster
cluster_no = -1
for cluster in  performance_models_per_cluster:
    cluster_no += 1

    print ">>>>>>>>>> Inside Cluster {} <<<<<<<<<<".format(cluster_no)

    performance_features = pd.DataFrame(cluster.keys(), columns=['Level', 'Robot_Feedback', 'Prev_Score'])
    performance_values = pd.DataFrame(cluster.values(), columns=['Success_Rate'])

    NN = MLPRegressor(max_iter=5000, random_state=42)
    NN.fit(performance_features,performance_values.values.ravel())

    performance_prediction = NN.predict(performance_features)
    NN_mse = mean_squared_error(performance_values, performance_prediction)
    NN_rmse = np.sqrt(NN_mse)
    print "\tTraining Data: NN Root Mean Square Error = {}".format(NN_rmse)



    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111)


    ax1.plot(performance_values, c='b', label='Original Output')
    ax1.plot(performance_prediction, c='r', label='Predicted OutPut')
    ax1.set_xlabel('N Samples')
    ax1.set_ylabel('P(Success | Level, Robot Feedback, Prev. Score)')
    ax1.set_title('Cluster {0} : Training Accuracy (RMSE= {1:0.3f})'.format(cluster_no, NN_rmse))
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig('cluster{}_Training_Accuracy.png'.format(cluster_no))
    print "\tClose the figure in the background to continue ..."
    plt.show()

    scores = cross_val_score(NN, performance_features, performance_values.values.ravel(), scoring='neg_mean_squared_error', cv = 5)
    rsme_scores = np.sqrt(-scores)

    display_scores("NN (5 folds Cross Validation) - RMSE ", rsme_scores)

print "\nEnd of the Program"