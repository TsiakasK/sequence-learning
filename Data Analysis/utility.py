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
from numpy.linalg import inv

seed = np.random.RandomState(seed=3)


markers = ['o', 'v', 'h', 'H', 'o', 'v', 'h', 'H', 'h', 'H', 'o', 'v', 'h']
colors = ['b', 'r','g','c','y','m', 'b', 'r','g','c','y','m','y','m', 'b']
combs = list(itertools.product(markers, colors))



dirname = 'clean_data'
users = os.listdir(dirname)
users_augmented_to_sessions = []
sessions_per_user = []
#D = np.zeros((len(pids), 24))
P = {}
D = []
user_models = []
user_utility_values = []
grading_system = [1,2,3,4]
utility1 = []
utility2 = []
utility3 = []
for user in users: 
    sessions = os.listdir(dirname + '/' + user)
    NoOfSessions = len(sessions)    
    for session in sessions:
        users_augmented_to_sessions.append(user)
        sessions_per_user.append(session)
        filename = dirname + '/' + user + '/' + session + '/logfile'
        f = open(filename, 'r')
        lines = f.readlines()
        f.close()
        P = {}
        scores = []
        #print user + '/' + session
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
         #   print P[p] 
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
      #  print scores
        for i in [1,2,3,4]: 
            um.append(scores.count(i)/float((scores.count(i) + scores.count(-1*i))))
        #print um
        user_models.append(um)

        #Unitility estimation 
        #print "UM:",user_models[-1]

        #utility1 estimation (if WIN: SCORE=[1,4], if LOSE: SCORE=0) U=Pu(S/Level)*[1,4])
        utility1.append([a*b for a,b in zip(user_models[-1],grading_system)])
            
        #utility2 estimation (if WIN: SCORE=[1,4], if LOSE: SCORE=[-4,-1]]) U=Pu(Success/Level)*[1,4]+Pu(Fail/Level)*[-4,-1]
     #   print  "user model:", [1-m for m in user_models[-1]]
      #  print
       # print "utility 1:", utility1[-1]
        #print "negative score:",[s * -1 for s in grading_system]
        #print "negative term:",[a*b for a,b in zip([1-m for m in user_models[-1]],[s * -1 for s in grading_system])]
        #print
        #"final matrix:", zip(utility1, [a*b for a,b in zip([1-m for m in user_models[-1]],[s * -1 for s in grading_system])])
        utility2.append([sum(x) for x in zip(utility1[-1], [a*b for a,b in zip([1-m for m in user_models[-1]],[s * -1 for s in grading_system])])])
        #utility3 estimation (if WIN: SCORE=[1,4], if LOSE: SCORE=[-4,-1]]) U=Pu(Success/Level)*[1,4]+Pu(Fail/Level)*[-4,-1]
        utility3.append([sum(x) for x in zip(utility1[-1], [a*b for a,b in zip([1-m for m in user_models[-1]],[-1/float(s)for s in grading_system] )])])
'''
print "USER MODELS"
print user_models
print "##################"
print "UTILITY-1 POSITIVE ONLY"
print utility1     
print "##################"
print "UTILITY-2 POSITIVE+NEGATIVE"
print utility2         
print "##################"
print "UTILITY-3 POSITIVE+NEGATIVE"
print utility3     
'''
'''
for i in range((len(utility1))):
    print "User"+str(i) +" user model:",user_models[i]
    print "\tuserMeanUtil\tminUtil\tmaxUtil\tminUtil\tmeanUtil\tstdUtil"
    print "u1:", np.mean(utility1[i]), min([np.mean(u) for u in utility1]),max([np.mean(u) for u in utility1]), np.mean([np.mean(u) for u in utility1]),np.std([np.mean(u) for u in utility1])
    print "u2:",np.mean(utility2[i]),min([np.mean(u) for u in utility2]),max([np.mean(u) for u in utility2]), np.mean([np.mean(u) for u in utility2]),np.std([np.mean(u) for u in utility2])
    print "u3:",np.mean(utility3[i]), min([np.mean(u) for u in utility3]),max([np.mean(u) for u in utility3]), np.mean([np.mean(u) for u in utility3]),np.std([np.mean(u) for u in utility3])
    print "#########################" 
    print   
'''




#PLOTS

plt.subplot(1,3,1)
user_means1 =  np.mean(np.array(utility1),axis=1)
max1 = np.max(user_means1)
min1 = np.min(user_means1)
mean1 = np.mean(user_means1)
std1 = np.std(user_means1)

user_means1 = [x for x,_ in sorted(zip(user_means1,users_augmented_to_sessions))]
users1 = [x for _,x in sorted(zip(user_means1,users_augmented_to_sessions))]
sessions1 = [x for _,x in sorted(zip(user_means1,sessions_per_user))]
users1 = [x for _,x in sorted(zip(user_means1,users_augmented_to_sessions))]

labels=[]

for i in range(len(users1)):
     x = users1[i].split('_')[-1]
     y = sessions1[i].split('_')[-1]
     labels.append(x+'_'+y)



#line = np.linspace(min1, max1, len(user_means1))
plt.plot( user_means1,[0]*len(user_means1))
#plt.hold(True)
plt.plot(user_means1,[0]*len(user_means1),'ro')
plt.xticks(user_means1, labels)



plt.subplot(1,3,2)

user_means2 =  np.mean(np.array(utility2),axis=1)
max2 = np.max(user_means2)
min2 = np.min(user_means2)
mean2 = np.mean(user_means2)
std2 = np.std(user_means2)


user_means2 = [x for x,_ in sorted(zip(user_means2,users_augmented_to_sessions))]
users2 = [x for _,x in sorted(zip(user_means2,users_augmented_to_sessions))]
sessions2 = [x for _,x in sorted(zip(user_means2,sessions_per_user))]
users2 = [x for _,x in sorted(zip(user_means2,users_augmented_to_sessions))]

labels=[]

for i in range(len(users2)):
     x = users2[i].split('_')[-1]
     y = sessions2[i].split('_')[-1]
     labels.append(x+'_'+y)


#line = np.linspace(min1, max1, len(user_means1))
plt.plot( user_means2,[0]*len(user_means2))
#plt.hold(True)
plt.plot(user_means2,[0]*len(user_means2),'ro')
plt.xticks(user_means2, labels)




plt.subplot(1,3,3)


user_means3 =  np.mean(np.array(utility3),axis=1)
max3 = np.max(user_means3)
min3 = np.min(user_means3)
mean3 = np.mean(user_means3)
std3 = np.std(user_means3)

user_means3 = [x for x,_ in sorted(zip(user_means3,users_augmented_to_sessions))]
users3 = [x for _,x in sorted(zip(user_means3,users_augmented_to_sessions))]
sessions3 = [x for _,x in sorted(zip(user_means3,sessions_per_user))]
users3 = [x for _,x in sorted(zip(user_means3,users_augmented_to_sessions))]

labels=[]

for i in range(len(users3)):
     x = users3[i].split('_')[-1]
     y = sessions3[i].split('_')[-1]
     labels.append(x+'_'+y)


plt.plot( user_means3,[0]*len(user_means3))
#plt.hold(True)
plt.plot(user_means3,[0]*len(user_means3),'ro')
plt.xticks(user_means3, users3)


plt.show()

with open("user_skills","w") as f:
    for i in range(len(users_augmented_to_sessions)):
          m1 = user_means1[users1.index(users_augmented_to_sessions[i])]
          m2 = user_means2[users2.index(users_augmented_to_sessions[i])]
          m3 = user_means3[users3.index(users_augmented_to_sessions[i])]
          f.write(users_augmented_to_sessions[i]+"/"+sessions_per_user[i]+" "+str(m1)+" "+str(m2)+" "+str(m3)+"\n")
f.close()



'''
print 
print users1
print
print users2
print 
print users3
'''