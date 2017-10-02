#!/usr/bin/python
import os
import sys
import matplotlib.pyplot as plt
import numpy as np 
import matplotlib.animation as animation
import re
import pandas as pd
from collections import Counter

FILE = 4


data1 = pd.read_csv('datasets/engagement_means.csv', delimiter=' ')				#1
data2 = pd.read_csv('datasets/engagement_means_normed_range.csv', delimiter=' ')		#2
data3 = pd.read_csv('datasets/engagement_means_normed_mean.csv', delimiter=' ')		#3
data4 = pd.read_csv('datasets/engagement_means_normed_mean_range.csv', delimiter=' ')		#4

DATA = [data1, data2, data3, data4]


#filename.write("ID L1a L2a L3a L4a L1b L2b L3b L4b L1c L2a L3c L4c L1d L2d L3d L4d\n")
for i, data in enumerate(DATA): 
	filename = open('datasets/mean_eng_per_level_' + str(i+1) + '.csv','w')
	users = data['ID'].unique()
	for u in users: 
		U = data.loc[data['ID'] == u]
		filename.write(u)
		for l in [3,5,7,9]: 
			filename.write(' '+ str(U.loc[U['length'] == l]["engagement"].mean()))
		filename.write('\n')
		



