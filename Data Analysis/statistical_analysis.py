#!/usr/bin/python
import os
import sys
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from pandas.tools.plotting import scatter_matrix
from scipy.stats import ttest_ind
import re

# preprocessing
names = ['cluster','engagement', 'length', 'current_result', 'previous_result', 'action']
f = open('logfiles/state_action_engagement.csv', 'r')
lines = f.readlines()
data = pd.read_csv('logfiles/state_action_engagement.csv', delimiter=' ')
data[['previous_result', 'current_result']] = data[['previous_result', 'current_result']].astype('float64') 

# cluster data
C0 = data.loc[data['cluster']==0][['engagement', 'length', 'current_result', 'previous_result', 'action']]
C1 = data.loc[data['cluster']==1][['engagement', 'length', 'current_result', 'previous_result', 'action']]
C2 = data.loc[data['cluster']==2][['engagement', 'length', 'current_result', 'previous_result', 'action']]

# engagement over failure - over all users
print "t-test for mean engagement over failure and success"
E0 = data.loc[data['current_result']==-1][['engagement']]
E1 = data.loc[data['current_result']==1][['engagement']]

print E0.mean(), E0.var()
print E1.mean(), E1.var()
print ttest_ind(E0, E1)

# engagement over failure - over clusters
print "t-test for mean engagement over failure and success over clusters"

print "cluster0"
C0E0 = C0.loc[data['current_result']==-1][['engagement']]
C0E1 = C0.loc[data['current_result']==1][['engagement']]
print C0E0.mean(), C0E0.var()
print C0E1.mean(), C0E1.var()
print ttest_ind(C0E0, C0E1)

print "cluster1"
C1E0 = C1.loc[data['current_result']==-1][['engagement']]
C1E1 = C1.loc[data['current_result']==1][['engagement']]
print C1E0.mean(), C1E0.var()
print C1E1.mean(), C1E1.var()
print ttest_ind(C1E0, C1E1)

print "cluster2"
C2E0 = C2.loc[data['current_result']==-1][['engagement']]
C2E1 = C2.loc[data['current_result']==1][['engagement']]
print C2E0.mean(), C2E0.var()
print C2E1.mean(), C2E1.var()
print ttest_ind(C2E0, C2E1)

print "engagement for cluster0 vs cluster2 over success"
print C0E1.mean(), C0E1.var()
print C2E1.mean(), C2E1.var()
print ttest_ind(C0E1, C2E1)

print "engagement for cluster0 vs cluster2 over failure"
print C0E0.mean(), C0E0.var()
print C2E0.mean(), C2E0.var()
print ttest_ind(C0E0, C2E0)

## cluster 0,1,2 on levels 1,2,3,4 over win/loss
# http://hamelg.blogspot.com/2015/11/python-for-data-analysis-part-24.html




