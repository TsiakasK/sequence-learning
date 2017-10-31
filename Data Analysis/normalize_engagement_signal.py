#!/usr/bin/python
import os
import sys
import matplotlib.pyplot as plt
import numpy as np 
import matplotlib.animation as animation
import re
from sklearn.preprocessing import scale


def normalize_by_mean(x): 
	x = np.asarray(x)
	return x-x.mean()

def normalize_by_range(x, nmin = 0, nmax = 1):
	x = np.asarray(x)
	return (nmax-nmin)*(x-min(x))/(max(x)-min(x)) + nmin

def normalize_by_mean_std(x): 
	x = np.asarray(x)
	return scale(x, axis=0, with_mean=True, with_std=True, copy=True)

def statistics(x):
	x = np.asarray(x)
	return min(x), max(x), x.mean()

efile = open('datasets/engagement_relative','r')
lines = efile.readlines()
efile.close()

current_user = ' '
user = []
full = []
MAX, MIN, MEAN = [], [], [] 
maxs, mins, means = [], [], []
sessions = []
sess = 0
U, C, L, RF, PS, CR, CS, points, A = [], [], [], [], [], [], [], [], []
for line in lines:
	a = line.split()
	user_session = a[0]
	cluster = a[1]
	length = a[2]
	rf = a[3]
	ps = a[4]
	cr = a[5]
	cs = a[6]
	eng = a[6:-1]
	action = a[-1]

	U.append(user_session)
	C.append(cluster)
	L.append(length)
	RF.append(rf)
	PS.append(ps)
	CS.append(cs)
	CR.append(cr)
	A.append(action)
	points.append(len(eng))
	
	if current_user != user_session:
		if user: 
			# signal statistics
			xmin, xmax, xmean = statistics(user) 			

			# normalization
			x1 = normalize_by_range(user)
			x2 = normalize_by_mean(user)
			x3 = normalize_by_range(x2, -1, 1)

			
			mins.append(xmin)
			maxs.append(xmax)
			user = []
	
		current_user = user_session
	
	#print "appending user"
	sessions.append(current_user) 
	sess += 1
	for e in eng: 
		user.append(float(e))
		full.append(float(e))
	means.append(np.asarray(eng).astype(float).mean())
#xnormed, xmax, xmin, xmean = user_statistics(full, 1, 1)
# signal statistics
xmin, xmax, xmean = statistics(full) 			

# normalization
x1 = normalize_by_range(full)
x2 = normalize_by_mean(full)
x3 = normalize_by_range(x2, -1, 1)
x4 = normalize_by_mean_std(full)

print means
weights = np.ones_like(means)/float(len(means))
plt.hist(means, weights = weights)
plt.title('Mean engagement per task turn')
plt.savefig('mean_engagement_per_turn_all_users.png')
"""
plt.hist(x1)
plt.show()
plt.hist(x2)
plt.show()
plt.hist(x3)
plt.show()
plt.hist(x4)
plt.show()
"""

# write to files
f = open('datasets/engagement_means.csv','w')
f1 = open('datasets/engagement_means_normed_range.csv','w')
f2 = open('datasets/engagement_means_normed_mean.csv','w')
f3 = open('datasets/engagement_means_normed_mean_range.csv','w')
f4 = open('datasets/engagement_scaled.csv','w')

f.write('ID cluster length robot_feedback previous_score current_score current_result action engagement_mean engagement_std\n')
f1.write('ID cluster length robot_feedback previous_score current_score current_result action engagement_mean engagement_std\n')
f2.write('ID cluster length robot_feedback previous_score current_score current_result action engagement_mean engagement_std\n')
f3.write('ID cluster length robot_feedback previous_score current_score current_result action engagement_mean engagement_std\n')
f4.write('ID cluster length robot_feedback previous_score current_score current_result action engagement_mean engagement_std\n')

""" varun
f.write('user_session_ID clusterID length robot_feedback previous_score current_result mean_engagement action\n')
f1.write('user_session_ID clusterID length robot_feedback previous_score current_result mean_engagement action\n')
f2.write('user_session_ID clusterID length robot_feedback previous_score current_result mean_engagement action\n')
f3.write('user_session_ID clusterID length robot_feedback previous_score current_result mean_engagement action\n')
"""

i = 0 
for u, c, l, r, p, cr, cs, e, a in zip(U, C, L, RF, PS, CR, CS, points, A): 
	m = np.asarray(full)[i:i+e].mean()
	s = np.asarray(full)[i:i+e].std()
	m1 = x1[i:i+e].mean()
	m2 = x2[i:i+e].mean()
	m3 = x3[i:i+e].mean()
	m4 = x4[i:i+e].mean()
	s1 = x1[i:i+e].std()
	s2 = x2[i:i+e].std()
	s3 = x3[i:i+e].std()
	s4 = x4[i:i+e].std()
	
	f.write(u + ' ' + str(c) + ' ' + str(l) + ' ' + str(r) + ' ' + str(p) + ' '  + str(cs) + ' ' + str(cr) + ' ' + str(a) + ' ' + str(m) + ' ' + str(s) +'\n')
	f1.write(u + ' ' + str(c) + ' ' + str(l) + ' ' + str(r) + ' ' + str(p) + ' ' + str(cs) + ' ' + str(cr) + ' ' + str(a) + ' ' + str(m1) + ' ' + str(s1) + '\n')
	f2.write(u + ' ' + str(c) + ' ' + str(l) + ' ' + str(r) + ' ' + str(p) + ' ' + str(cs) + ' ' + str(cr) + ' ' + str(a) + ' ' + str(m2) + ' ' + str(s2) + '\n')
	f3.write(u + ' ' + str(c) + ' ' + str(l) + ' ' + str(r) + ' ' + str(p) + ' ' + str(cs) + ' ' + str(cr) + ' ' + str(a) + ' ' + str(m3) + ' ' + str(s3) +'\n')
	f4.write(u + ' ' + str(c) + ' ' + str(l) + ' ' + str(r) + ' ' + str(p) + ' ' + str(cs) + ' ' + str(cr) + ' ' + str(a) + ' ' + str(m4) + ' ' + str(s4) + '\n')
	
	""" varun
	f.write(u + ' ' +  str(c) + ' ' + str(l) + ' ' + str(r) + ' ' + str(p) + ' ' + str(cr) + ' ' + str(x) + ' ' + str(a) + '\n')
	f1.write(u + ' ' + str(c) + ' ' + str(l) + ' ' + str(r) + ' ' + str(p) + ' ' + str(cr) + ' ' + str(s1) + ' ' + str(a) + '\n')
	f2.write(u + ' ' + str(c) + ' ' + str(l) + ' ' + str(r) + ' ' + str(p) + ' ' + str(cr) + ' ' + str(s2) + ' ' + str(a) + '\n')
	f3.write(u + ' ' + str(c) + ' ' + str(l) + ' ' + str(r) + ' ' + str(p) + ' ' + str(cr) + ' ' + str(s3) + ' ' + str(a) + '\n')
	"""

	i += e

"""
# plots and histograms 
plt.subplot(241)
plt.plot(full)
plt.title('Original - Mean = ' + str(xmean))

plt.subplot(242)
plt.plot(x1)
plt.title('range - Mean = ' + str(x1.mean()))

plt.subplot(243)
plt.plot(x2)
plt.title('mean - Mean = ' + str(x2.mean()))

plt.subplot(244)
plt.plot(x3)
plt.title('mean-range - Mean = ' + str(x2.mean()))

plt.subplot(245)
weights = 100*np.ones_like(full)/len(full)
plt.hist(full, weights = weights)

plt.subplot(246)
weights = 100*np.ones_like(x1)/len(x1)
plt.hist(x1, weights = weights)

plt.subplot(247)
weights = 100*np.ones_like(x2)/len(x2)
plt.hist(x2, weights = weights)

plt.subplot(248)
weights = 100*np.ones_like(x3)/len(x3)
plt.hist(x3, weights = weights)
plt.show()
"""
