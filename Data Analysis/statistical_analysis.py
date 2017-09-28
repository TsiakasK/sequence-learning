#!/usr/bin/python
import os
import sys
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from pandas.tools.plotting import scatter_matrix
#import statsmodels.api as sm
#from statsmodels.formula.api import ols
import re
import seaborn as sns

def correlation_matrix(df):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    #plt.title('Abalone Feature Correlation')
    labels=['engagement', 'length', 'current_result']
    ax1.set_xticklabels(labels,fontsize=6)
    ax1.set_yticklabels(labels,fontsize=6)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])
    plt.show()

names = ['engagement', 'length', 'current_result']
f = open('logfiles/state_action_engagement.csv', 'r')
lines = f.readlines()
data = pd.read_csv('logfiles/state_action_engagement.csv', delimiter=' ')
data[['previous_result', 'current_result']] = data[['previous_result', 'current_result']].astype('float64') 
C1 = data.loc[data['cluster']==1]
data = data[['engagement', 'length', 'current_result', 'previous_result']]
print C1
correlations = data.corr()
print correlations

# plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations)
fig.colorbar(cax)
#ticks = numpy.arange(0,9,1)
#ax.set_xticks(ticks)
#ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()
