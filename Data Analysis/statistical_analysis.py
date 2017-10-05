#!/usr/bin/python
import os
import sys
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from pandas.tools.plotting import scatter_matrix
from scipy.stats import ttest_ind
import re

# datasets
data = pd.read_csv('datasets/engagement_means_normed_range.csv', delimiter=' ')
#data['engagement'].hist()
#plt.title('Normalized Engagement Means Histogram')
#plt.show()

# all user data
C = data[['cluster','engagement', 'length', 'previous_score', 'current_score' , 'current_result', 'action']]

# cluster data
C0 = C.loc[C['cluster']==0]
C1 = C.loc[C['cluster']==1]
C2 = C.loc[C['cluster']==2]
C3 = C.loc[C['cluster']==3]
#C1 = pd.concat([C1, C2])

# performance-based grouping - over all users
P0 = C.loc[C['current_result']==-1]
P1 = C.loc[C['current_result']==1]

# performance-based grouping - over clusters
C0P0 = C0.loc[C0['current_result']==-1]
C0P1 = C0.loc[C0['current_result']==1]

C1P0 = C1.loc[C1['current_result']==-1]
C1P1 = C1.loc[C1['current_result']==1]

C2P0 = C2.loc[C2['current_result']==-1]
C2P1 = C2.loc[C2['current_result']==1]

# level grouping -- all users
L1 = C.loc[C['length']==3]
L2 = C.loc[C['length']==5]
L3 = C.loc[C['length']==7]
L4 = C.loc[C['length']==9]

# level grouping -- per performance
L1P0 = L1.loc[L1['current_result']==-1]
L1P1 = L1.loc[L1['current_result']==1]

L2P0 = L2.loc[L2['current_result']==-1]
L2P1 = L2.loc[L2['current_result']==1]

L3P0 = L3.loc[L3['current_result']==-1]
L3P1 = L3.loc[L3['current_result']==1]

L4P0 = L4.loc[L4['current_result']==-1]
L4P1 = L4.loc[L4['current_result']==1]

# level grouping -- per clusters
C0L1 = C0.loc[C0['length']==3]
C0L2 = C0.loc[C0['length']==5]
C0L3 = C0.loc[C0['length']==7]
C0L4 = C0.loc[C0['length']==9]

C1L1 = C1.loc[C1['length']==3]
C1L2 = C1.loc[C1['length']==5]
C1L3 = C1.loc[C1['length']==7]
C1L4 = C1.loc[C1['length']==9]

C2L1 = C2.loc[C2['length']==3]
C2L2 = C2.loc[C2['length']==5]
C2L3 = C2.loc[C2['length']==7]
C2L4 = C2.loc[C2['length']==9]

C3L1 = C3.loc[C3['length']==3]
C3L2 = C3.loc[C3['length']==5]
C3L3 = C3.loc[C3['length']==7]
C3L4 = C3.loc[C3['length']==9]

# level grouping -- per clusters and performance
C0L1P0 = L1P0.loc[L1P0['cluster']==0]
C0L1P1 = L1P1.loc[L1P1['cluster']==0]
C0L2P0 = L2P0.loc[L2P0['cluster']==0]
C0L2P1 = L2P1.loc[L2P1['cluster']==0]
C0L3P0 = L3P0.loc[L3P0['cluster']==0]
C0L3P1 = L3P1.loc[L3P1['cluster']==0]
C0L4P0 = L4P0.loc[L4P0['cluster']==0]
C0L4P1 = L4P1.loc[L4P1['cluster']==0]

C1L1P0 = L1P0.loc[L1P0['cluster']==1]
C1L1P1 = L1P1.loc[L1P1['cluster']==1]
C1L2P0 = L2P0.loc[L2P0['cluster']==1]
C1L2P1 = L2P1.loc[L2P1['cluster']==1]
C1L3P0 = L3P0.loc[L3P0['cluster']==1]
C1L3P1 = L3P1.loc[L3P1['cluster']==1]
C1L4P0 = L4P0.loc[L4P0['cluster']==1]
C1L4P1 = L4P1.loc[L4P1['cluster']==1]

C2L1P0 = L1P0.loc[L1P0['cluster']==2]
C2L1P1 = L1P1.loc[L1P1['cluster']==2]
C2L2P0 = L2P0.loc[L2P0['cluster']==2]
C2L2P1 = L2P1.loc[L2P1['cluster']==2]
C2L3P0 = L3P0.loc[L3P0['cluster']==2]
C2L3P1 = L3P1.loc[L3P1['cluster']==2]
C2L4P0 = L4P0.loc[L4P0['cluster']==2]
C2L4P1 = L4P1.loc[L4P1['cluster']==2]

C3L1P0 = L1P0.loc[L1P0['cluster']==3]
C3L1P1 = L1P1.loc[L1P1['cluster']==3]
C3L2P0 = L2P0.loc[L2P0['cluster']==3]
C3L2P1 = L2P1.loc[L2P1['cluster']==3]
C3L3P0 = L3P0.loc[L3P0['cluster']==3]
C3L3P1 = L3P1.loc[L3P1['cluster']==3]
C3L4P0 = L4P0.loc[L4P0['cluster']==3]
C3L4P1 = L4P1.loc[L4P1['cluster']==3]


# correlation of performance and engagement 
print  C[['length', 'current_result']].corr()
print  C0[['length', 'current_result']].corr()
print  C1[['length', 'current_result']].corr()
print  C2[['length', 'current_result']].corr()

# mean engagements over sussess/failure
print "mean engagement over performance"
print ttest_ind(P0[['engagement']], P1[['engagement']])


# mean engagements over sussess/failure on each level 
print "mean engagement over performance in all levels"
print ttest_ind(L1P0[['engagement']], L1P1[['engagement']])
print ttest_ind(L2P0[['engagement']], L2P1[['engagement']])
print ttest_ind(L3P0[['engagement']], L3P1[['engagement']])
print ttest_ind(L4P0[['engagement']], L4P1[['engagement']])
print 
# mean engagements over levels
print "mean engagement in all levels C0-C1"
print ttest_ind(C0L1[['engagement']], C1L1[['engagement']])
print ttest_ind(C0L2[['engagement']], C1L2[['engagement']])
print ttest_ind(C0L3[['engagement']], C1L3[['engagement']])
print ttest_ind(C0L4[['engagement']], C1L4[['engagement']])
print 

# mean engagements over levels
print "mean engagement in all levels C0-C2"
print ttest_ind(C0L1[['engagement']], C2L1[['engagement']])
print ttest_ind(C0L2[['engagement']], C2L2[['engagement']])
print ttest_ind(C0L3[['engagement']], C2L3[['engagement']])
print ttest_ind(C0L4[['engagement']], C2L4[['engagement']])
print
# mean engagements over levels
print "mean engagement in all levels C1-C2"
print ttest_ind(C1L1[['engagement']], C2L1[['engagement']])
print ttest_ind(C1L2[['engagement']], C2L2[['engagement']])
print ttest_ind(C1L3[['engagement']], C2L3[['engagement']])
print ttest_ind(C1L4[['engagement']], C2L4[['engagement']])
print
"""
# mean engagements over levels
print "mean engagement in all levels C1-C3"
print ttest_ind(C1L1[['engagement']], C3L1[['engagement']])
print ttest_ind(C1L2[['engagement']], C3L2[['engagement']])
print ttest_ind(C1L3[['engagement']], C3L3[['engagement']])
print ttest_ind(C1L4[['engagement']], C3L4[['engagement']])
print

# mean engagements over levels
print "mean engagement in all levels C2-C3"
print ttest_ind(C2L1[['engagement']], C3L1[['engagement']])
print ttest_ind(C2L2[['engagement']], C3L2[['engagement']])
print ttest_ind(C2L3[['engagement']], C3L3[['engagement']])
print ttest_ind(C2L4[['engagement']], C3L4[['engagement']])
print
"""

# engagement over success/failure for each cluster
print "cluster 1 - mean engagement in all levels over failure/success"
print ttest_ind(C0L1P0[['engagement']], C0L1P1[['engagement']])
print ttest_ind(C0L2P0[['engagement']], C0L2P1[['engagement']])
print ttest_ind(C0L3P0[['engagement']], C0L3P1[['engagement']])
print ttest_ind(C0L4P0[['engagement']], C0L4P1[['engagement']])

print "cluster 2 - mean engagement in all levels over failure/success"
print ttest_ind(C1L1P0[['engagement']], C1L1P1[['engagement']])
print ttest_ind(C1L2P0[['engagement']], C1L2P1[['engagement']])
print ttest_ind(C1L3P0[['engagement']], C1L3P1[['engagement']])
print ttest_ind(C1L4P0[['engagement']], C1L4P1[['engagement']])

print "cluster 3 - mean engagement in all levels over failure/success"
print ttest_ind(C2L1P0[['engagement']], C2L1P1[['engagement']])
print ttest_ind(C2L2P0[['engagement']], C2L2P1[['engagement']])
print ttest_ind(C2L3P0[['engagement']], C2L3P1[['engagement']])
print ttest_ind(C2L4P0[['engagement']], C2L4P1[['engagement']])

"""
print "cluster 4 - mean engagement in all levels over failure/success"
print ttest_ind(C0L1P0[['engagement']], C0L1P1[['engagement']])
print ttest_ind(C0L2P0[['engagement']], C0L2P1[['engagement']])
print ttest_ind(C0L3P0[['engagement']], C0L3P1[['engagement']])
print ttest_ind(C0L4P0[['engagement']], C0L4P1[['engagement']])
"""
print 

mean_c0l1 =  C0L1[['engagement']].mean()[0]
mean_c0l2 =  C0L2[['engagement']].mean()[0]
mean_c0l3 =  C0L3[['engagement']].mean()[0]
mean_c0l4 =  C0L4[['engagement']].mean()[0]
var_c0l1 =  C0L1[['engagement']].var()[0]
var_c0l2 =  C0L2[['engagement']].var()[0]
var_c0l3 =  C0L3[['engagement']].var()[0]
var_c0l4 =  C0L4[['engagement']].var()[0]
C0_mean = [mean_c0l1, mean_c0l2, mean_c0l3, mean_c0l4]
C0_var = [var_c0l1, var_c0l2, var_c0l3, var_c0l4]

mean_c1l1 =  C1L1[['engagement']].mean()[0]
mean_c1l2 =  C1L2[['engagement']].mean()[0]
mean_c1l3 =  C1L3[['engagement']].mean()[0]
mean_c1l4 =  C1L4[['engagement']].mean()[0]
var_c1l1 =  C1L1[['engagement']].var()[0]
var_c1l2 =  C1L2[['engagement']].var()[0]
var_c1l3 =  C1L3[['engagement']].var()[0]
var_c1l4 =  C1L4[['engagement']].var()[0]
C1_mean = [mean_c1l1, mean_c1l2, mean_c1l3, mean_c1l4]
C1_var = [var_c1l1, var_c1l2, var_c1l3, var_c1l4]

mean_c2l1 =  C2L1[['engagement']].mean()[0]
mean_c2l2 =  C2L2[['engagement']].mean()[0]
mean_c2l3 =  C2L3[['engagement']].mean()[0]
mean_c2l4 =  C2L4[['engagement']].mean()[0]
var_c2l1 =  C2L1[['engagement']].var()[0]
var_c2l2 =  C2L2[['engagement']].var()[0]
var_c2l3 =  C2L3[['engagement']].var()[0]
var_c2l4 =  C2L4[['engagement']].var()[0]
C2_mean = [mean_c2l1, mean_c2l2, mean_c2l3, mean_c2l4]
C2_var = [var_c2l1, var_c2l4, var_c2l3, var_c2l4]

mean_c3l1 =  C3L1[['engagement']].mean()[0]
mean_c3l2 =  C3L2[['engagement']].mean()[0]
mean_c3l3 =  C3L3[['engagement']].mean()[0]
mean_c3l4 =  C3L4[['engagement']].mean()[0]
var_c3l1 =  C3L1[['engagement']].var()[0]
var_c3l2 =  C3L2[['engagement']].var()[0]
var_c3l3 =  C3L3[['engagement']].var()[0]
var_c3l4 =  C3L4[['engagement']].var()[0]
C3_mean = [mean_c3l1, mean_c3l2, mean_c3l3, mean_c3l4]
C3_var = [var_c3l1, var_c3l2, var_c3l3, var_c3l4]

# overall 
mean_l1 =  L1[['engagement']].mean()[0]
mean_l2 =  L2[['engagement']].mean()[0]
mean_l3 =  L3[['engagement']].mean()[0]
mean_l4 =  L4[['engagement']].mean()[0]
var_l1 =  L1[['engagement']].var()[0]
var_l2 =  L2[['engagement']].var()[0]
var_l3 =  L3[['engagement']].var()[0]
var_l4 =  L4[['engagement']].var()[0]
C_mean = [mean_l1, mean_l2, mean_l3, mean_l4]
C_var = [var_l1, var_l2, var_l3, var_l4]

X = [1,2,3,4]
labels = ['Level 1', 'Level 2', 'Level 3', 'Level 4']
plt.errorbar(X, C0_mean, yerr = C0_var, fmt='o--', color = 'b', ecolor='b', label = 'cluster_1')
plt.hold(True)
plt.errorbar(X, C1_mean, yerr = C0_var, fmt='o--', color = 'r', ecolor='r', label = 'cluster_2')

if not C2.empty: 
	plt.hold(True)
	plt.errorbar(X, C2_mean, yerr = C0_var, color = 'g', ecolor='g', fmt='o--', label = 'cluster_3')
if not C3.empty: 
	plt.hold(True)
	plt.errorbar(X, C3_mean, yerr = C0_var, fmt='o--', color = 'c', ecolor='c', label = 'cluster_4')

plt.hold(True)
plt.errorbar(X, C_mean, yerr = C_var, fmt='o--', color = 'k', ecolor= 'k', label = 'all users')

plt.title("Mean Engagement per Level")
plt.ylabel('Mean Engagement')
plt.xticks(X, labels)
plt.xlim([0.5,4.5])
plt.legend(loc = 2, fontsize = 'medium')
plt.savefig('engagement_per_level.png')

# http://hamelg.blogspot.com/2015/11/python-for-data-analysis-part-24.html




