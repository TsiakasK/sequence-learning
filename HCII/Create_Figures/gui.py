#!/usr/bin/python
import numpy as np
import os
import re
import math
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
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from os import system


def columns2list(h):
    hs = [x.encode('UTF8') for x in h.tolist()]
    return  hs

def myGuess(y):
    random_rate = random.uniform(0, 1)
    if random_rate <= y:
        return 1
    else:
        return -1

def HUM_graph(X, Y):
    ax1.set_xlabel('Task Turn')
    ax1.set_ylabel('Performance')
    ax1.set_title('History Based Monitoring')

    # HUM 0
    #ax1.axis([0, len(Y)+0.5, -1.0, 1.0])

    # HUM 1
    ax1.axis([0, len(Y) + 0.5, 0.0, 1.0])


    ax1.grid('on')

    for line in ax1.get_xgridlines():
        line.set_linestyle(' ')
    for line in ax1.get_ygridlines():
        line.set_linestyle('-.')

    ax1.plot(np.zeros(len(Y)+5), color = 'k')

    ax1.bar(-4, 0.5, 0.5, color= myColors[0], label='Level 1')
    ax1.bar(-3, 0.5, 0.5, color=myColors[1], label='Level 2')
    ax1.bar(-2, 0.5, 0.5, color=myColors[2], label='Level 3')
    ax1.bar(-1, 0.5, 0.5, color=myColors[3], label='Level 4')
    for idx in range(0, len(Y)):
        #HUM 0
        #ax1.bar(idx+1, myGuess(Y[idx]), 0.9, color=myColors[X[idx]-1])#, label='Level ' + str(X[idx]))


        #HUM 1
        x = X[idx]
        y = Y[idx]
        current_prediction = myGuess(y)

        if performance_dict.has_key(x):
            performance_dict[x].append(current_prediction)
        else:
            performance_dict[x] = [current_prediction]

        current_level_data = performance_dict[x]
        success_rate = (1.0 * current_level_data.count(1)/len(current_level_data))
        ax1.bar(idx + 1, success_rate + 0.02 , 0.9, color=myColors[X[idx] - 1])

        #print 'Level {0} \t Y {1:0.2f} \t Pred {2} \t S_Rate {3:0.2f} '.format(x,y, current_prediction, success_rate)


def DUM0_graph(X, Y):


    #ax2.set_xlabel('Task Turn')
    ax2.axes.get_xaxis().set_visible(False)
    ax2.set_ylabel('Performance')
    ax2.set_title('DUM Based Monitoring')


    ax2.axis([0.5, 4.5, 0.0, 1.0])

    ax2.grid('on')
    gridlines = ax2.get_xgridlines() + ax2.get_ygridlines()
    for line in gridlines:
        line.set_linestyle('-.')

    ax2.plot(np.zeros(len(Y)+5), color = 'k')

    ax2.bar(-4, 0.5, 0.5, color= myColors[0], label='Level 1')
    ax2.bar(-3, 0.5, 0.5, color=myColors[1], label='Level 2')
    ax2.bar(-2, 0.5, 0.5, color=myColors[2], label='Level 3')
    ax2.bar(-1, 0.5, 0.5, color=myColors[3], label='Level 4')


    for l in range(1,5):
        current_level_data = performance_dict[l]
        success_rate = (1.0 * current_level_data.count(1) / len(current_level_data))
        ax2.bar(l, success_rate, 0.9, color=myColors[l - 1])


def DUM1_graph(X, Y):
    ax3.axes.get_xaxis().set_visible(False)
    ax3.set_ylabel('Performance')
    ax3.set_title('DUM Based Monitoring')

    ax3.axis([0.5, 4.5, 0.0, 1.0])

    ax3.grid('on')
    gridlines = ax3.get_xgridlines() + ax3.get_ygridlines()
    for line in gridlines:
        line.set_linestyle('-.')

    ax3.plot(np.zeros(len(Y) + 5), color='k')

    ax3.bar(-4, 0.5, 0.5, color=myColors[0], label='Level 1')
    ax3.bar(-3, 0.5, 0.5, color=myColors[1], label='Level 2')
    ax3.bar(-2, 0.5, 0.5, color=myColors[2], label='Level 3')
    ax3.bar(-1, 0.5, 0.5, color=myColors[3], label='Level 4')


    #Add Success
    for l in range(4,1,-1):
        performance_dict[l-1].append(0)

        performance_dict[l-1].append(sum([ i for i in performance_dict[l] if i > 0]))

    #Add Failure
    for l in range(1,4):
        performance_dict[l+1].append(0)

        performance_dict[l+1].append(sum([ i for i in performance_dict[l] if i < 0 ]))



    for l in range(1,5):
        print l
        print performance_dict[l]
        current_level_data = performance_dict[l]
        level_length = sum([abs(v) for v in current_level_data])
        positive_scores = sum([v for v in current_level_data if v>0])
        success_rate = (1.0 * positive_scores / level_length)
        ax3.bar(l, success_rate, 0.9, color=myColors[l - 1])


#        success_rate = (1.0 * current_level_data.count(1) / len(current_level_data))
        #ax2.bar(l, success_rate + 0.02, 0.9, color=myColors[l - 1])






clusters = []
performance_dict = {}
myColors=['gold','darkorange','red', 'maroon']

fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)

fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)

fig3 = plt.figure(3)
ax3 = fig3.add_subplot(111)

random.seed(4)


for cluster_i in range(2, 3):#3
    print '\t\t----------------- Cluster {} -----------------'.format(cluster_i)


    with open('Clusters Data/Cluster_{0}_Normalized_Performance_Data'.format(cluster_i), 'rb') as file:
        cluster_x_data = pickle.load(file)

    clusters.append(cluster_x_data)
    all_headers = columns2list(cluster_x_data.columns)
    levels = cluster_x_data[all_headers[0]].tolist()
    labels = cluster_x_data[all_headers[-1]].tolist()

    print '--------------------- HUM ---------------------'
    HUM_graph(levels, labels)
    print '--------------------- DUM 0 ---------------------'
    DUM0_graph(levels, labels)
    print '--------------------- DUM 1 ---------------------'
    DUM1_graph(levels, labels)



lgd1 = ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
                  fancybox=True, shadow=True, ncol=4)
fig1.savefig('HUM1.png', bbox_extra_artists=(lgd1,), bbox_inches='tight')


lgd2 = ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.02),
                  fancybox=True, shadow=True, ncol=4)
fig2.savefig('DUM0.png', bbox_extra_artists=(lgd2,), bbox_inches='tight')


lgd3 = ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.02),
                  fancybox=True, shadow=True, ncol=4)
fig3.savefig('DUM1.png', bbox_extra_artists=(lgd3,), bbox_inches='tight')


system('say GUI.py is complete')
print 'End of the Program'

