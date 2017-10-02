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
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor


def normalize_DataFrame(dt):
    dt['Level'] = dt['Level'] / 3.0
    dt['Robot_Feedback'] = dt['Robot_Feedback'] / 2.0
    dt['Prev_Score'] = (dt['Prev_Score']  + 4)/8.0
    return dt

def actual_label(xFeatures, all_features, all_labels):
    #xFeatures is a Pandas Series in the form of [index, Level, Robot_Feedback, Prev_Score]
    data_row = all_features.loc[(all_features['Level'] == xFeatures[2])  & (all_features['Robot_Feedback'] == xFeatures[2] ) & (all_features['Prev_Score'] == xFeatures[3])]

    if data_row.empty:
        return -1.0
    else:
        return all_labels.loc[data_row.index.values[0]].values[0]
"""
def test_NN():
    #The code used to find the best NN
    min1 = 1
    min2 = 1
    for x in range(2,16):
        for y in range(2, 16):
            for solv in [  'sgd','lbfgs', 'adam']:
                # NN Model
                NN = MLPRegressor(activation='logistic', max_iter=5000, solver=solv, hidden_layer_sizes=(x, y), random_state=42)
                NN.fit(strat_train_set, train_labels.values.ravel())

                # Training Accuracy
                NN_prediction = NN.predict(strat_train_set)
                NN_mse = mean_squared_error(train_labels, NN_prediction)
                NN_rmse1 = np.sqrt(NN_mse)
                #print "\t\tTraining Data: NN Root Mean Square Error = {0:0.2f}".format(NN_rmse1)

                # Testing Accuracy
                NN_prediction = NN.predict(strat_test_set)
                NN_mse = mean_squared_error(test_labels, NN_prediction)
                NN_rmse2 = (np.sqrt(NN_mse))
                #print "\t\tTesting Data: NN Root Mean Square Error = {0:0.2f}".format(NN_rmse2)

                if min1 > NN_rmse1:
                    min1 = NN_rmse1
                    print '({0} , {1}) - {2} [{3:0.2f} , {4:0.2f}]'.format(x,y,solv,NN_rmse1,NN_rmse2)

                if min2 > NN_rmse2:
                    min2 = NN_rmse2
                    print '({0} , {1}) - {2} [{3:0.2f} , {4:0.2f}]'.format(x,y,solv,NN_rmse1,NN_rmse2)

def test_SVR():
# SVR
    min1 = 1
    min2 = 1
    for k in ['linear','poly','rbf','sigmoid']:
        for c in [0.1, 10.0, 100.0]:
            for e in [0.001,0.01, 0.1, 0.3, 0.7]:
                svr = svm.SVR(kernel=k, epsilon=e, tol=1e-3, C=c)
                svr.fit(strat_train_set, train_labels.values.ravel())

                # Training Accuracy
                svr_prediction = svr.predict(strat_train_set)
                svr_mse = mean_squared_error(train_labels, svr_prediction)
                svr_rmse1 = np.sqrt(svr_mse)
                #print "\t\tTraining Data: SVR Root Mean Square Error = {0:0.2f}".format(svr_rmse1)

                # Testing Accuracy
                svr_prediction = svr.predict(strat_test_set)
                svr_mse = mean_squared_error(test_labels, svr_prediction)
                svr_rmse2 = np.sqrt(svr_mse)
                #print "\t\tTesting Data: SVR Root Mean Square Error = {0:0.2f}".format(svr_rmse2)

                if min1 > svr_rmse1:
                    min1 = svr_rmse1
                    print '({0} , {1}) - {2} [{3:0.2f} , {4:0.2f}]'.format(c,e,k, svr_rmse1, svr_rmse2)

                if min2 > svr_rmse2:
                    min2 = svr_rmse2
                    print '({0} , {1}) - {2} [{3:0.2f} , {4:0.2f}]'.format(c,e,k, svr_rmse1, svr_rmse2)

def test_TR():
    min1 = 1
    min2 = 1

    for ss in [2,3,4,5,6,7]:
        for sl in [ 1, 2,3,4,5,6,7]:
            for wfl in [0.0, 0.1, 0.2, 0.3, 0.4]:
                # Decision Tree Regressor
                tree_reg = DecisionTreeRegressor(random_state=42, min_samples_split = ss, min_samples_leaf=sl, min_weight_fraction_leaf=wfl)
                tree_reg.fit(strat_train_set, train_labels.values.ravel())

                # Training Accuracy
                TR_prediction = tree_reg.predict(strat_train_set)
                TR_mse = mean_squared_error(train_labels, TR_prediction)
                TR_rmse1 = np.sqrt(TR_mse)
                #print "\t\tTraining Data: TR Root Mean Square Error = {0:0.2f}".format(TR_rmse1)

                # Testing Accuracy
                TR_prediction = tree_reg.predict(strat_test_set)
                TR_mse = mean_squared_error(test_labels, TR_prediction)
                TR_rmse2 = np.sqrt(TR_mse)
                #print "\t\tTesting Data: TR Root Mean Square Error = {0:0.2f}".format(TR_rmse2)

                if min1 >= TR_rmse1:
                    min1 = TR_rmse1
                    print '({0} , {1}) - {2} [{3:0.2f} , {4:0.2f}]'.format(ss,sl,wfl, TR_rmse1, TR_rmse2)

                if min2 >= TR_rmse2:
                    min2 = TR_rmse2
                    print '({0} , {1}) - {2} [{3:0.2f} , {4:0.2f}]'.format(ss,sl,wfl, TR_rmse1, TR_rmse2)

"""
dirname = '../../sequence-learning-master/Data Analysis/'
users = os.listdir(dirname)
data_files_per_cluster = {}
performance_models_per_cluster = []

# Read the clusters
clusters_file = dirname + 'datasets/user_clusters.csv'
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
            level = abs(int(a[3])) - 1  # 0 1 2 3
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

"""
with open('performance_models_per_cluster','w') as file:
    pickle.dump(performance_models_per_cluster,file)

print "Successfully saved Performance Models for {} Clusters!".format(len(performance_models_per_cluster))
"""

""" Regression Model using NN """


rfr_rmse2 = []
lsvr_rmse2 = []
nusvr_rmse2 = []
svr_rmse2 = []
TR_rmse2 = []
LR_rmse2 = []
NN_rmse2 = []


cluster_no = -1
for cluster in  performance_models_per_cluster:
    cluster_no += 1

    print ">>>>>>>>>> Inside Cluster {} <<<<<<<<<<".format(cluster_no)

    features = pd.DataFrame(cluster.keys(), columns=['Level', 'Robot_Feedback', 'Prev_Score'])

    norm_features = normalize_DataFrame(features.copy())

    labels = pd.DataFrame(cluster.values(), columns=['Success_Rate'])

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(norm_features, norm_features['Level']):
        strat_train_set = norm_features.loc[train_index]
        train_labels = labels.loc[train_index]

        strat_test_set = norm_features.loc[test_index]
        test_labels = labels.loc[test_index]
    #test_NN()
    #test_SVR()
    #test_TR()
    #continue

    """
    # Random Forest
    rfr = RandomForestRegressor( random_state=42, n_estimators=5, )
    rfr.fit(strat_train_set, train_labels.values.ravel())

    # Training Accuracy
    rfr_prediction = rfr.predict(strat_train_set)
    rfr_mse = mean_squared_error(train_labels, rfr_prediction)
    rfr_rmse1 = np.sqrt(rfr_mse)
    print "\t\tTraining Data: Random Forset Root Mean Square Error = {0:0.2f}".format(rfr_rmse1)

    # Testing Accuracy
    rfr_prediction = rfr.predict(strat_test_set)
    rfr_mse = mean_squared_error(test_labels, rfr_prediction)
    rfr_rmse2.append(np.sqrt(rfr_mse))
    print "\t\tTesting Data: RandomForestRoot Mean Square Error = {0:0.2f}".format(rfr_rmse2[-1])


    # Linear SVR
    lsvr = svm.LinearSVR( random_state=42, tol=1e-4, C=12.0)
    lsvr.fit(strat_train_set, train_labels.values.ravel())

    # Training Accuracy
    lsvr_prediction = lsvr.predict(strat_train_set)
    lsvr_mse = mean_squared_error(train_labels, lsvr_prediction)
    lsvr_rmse1 = np.sqrt(lsvr_mse)
    print "\t\tTraining Data: Linear SVR Root Mean Square Error = {0:0.2f}".format(lsvr_rmse1)

    # Testing Accuracy
    lsvr_prediction = lsvr.predict(strat_test_set)
    lsvr_mse = mean_squared_error(test_labels, lsvr_prediction)
    lsvr_rmse2.append(np.sqrt(lsvr_mse))
    print "\t\tTesting Data: Linear SVR Root Mean Square Error = {0:0.2f}".format(lsvr_rmse2[-1])



    # NuSVR
    nusvr = svm.NuSVR(kernel='linear', tol=1e-3, C=25.0, degree=5)
    nusvr.fit(strat_train_set, train_labels.values.ravel())

    # Training Accuracy
    nusvr_prediction = nusvr.predict(strat_train_set)
    nusvr_mse = mean_squared_error(train_labels, nusvr_prediction)
    nusvr_rmse1 = np.sqrt(nusvr_mse)
    print "\t\tTraining Data: NuSVR Root Mean Square Error = {0:0.2f}".format(nusvr_rmse1)

    # Testing Accuracy
    nusvr_prediction = nusvr.predict(strat_test_set)
    nusvr_mse = mean_squared_error(test_labels, nusvr_prediction)
    nusvr_rmse2.append(np.sqrt(nusvr_mse))
    print "\t\tTesting Data: NuSVR Root Mean Square Error = {0:0.2f}".format(nusvr_rmse2[-1])


    #SVR
    svr = svm.SVR(kernel='rbf', epsilon=0.01, C=10.0)
    svr.fit(strat_train_set, train_labels.values.ravel())

    # Training Accuracy
    svr_prediction = svr.predict(strat_train_set)
    svr_mse = mean_squared_error(train_labels, svr_prediction)
    svr_rmse1 = np.sqrt(svr_mse)
    print "\t\tTraining Data: SVR Root Mean Square Error = {0:0.2f}".format(svr_rmse1)

    # Testing Accuracy
    svr_prediction = svr.predict(strat_test_set)
    svr_mse = mean_squared_error(test_labels, svr_prediction)
    svr_rmse2.append(np.sqrt(svr_mse))
    print "\t\tTesting Data: SVR Root Mean Square Error = {0:0.2f}".format(svr_rmse2[-1])

    # Decision Tree Regressor
    tree_reg = DecisionTreeRegressor(random_state=42, min_samples_split = 2, min_samples_leaf=1, min_weight_fraction_leaf=0.2)
    tree_reg.fit(strat_train_set, train_labels.values.ravel())

    # Training Accuracy
    TR_prediction = tree_reg.predict(strat_train_set)
    TR_mse = mean_squared_error(train_labels, TR_prediction)
    TR_rmse1 = np.sqrt(TR_mse)
    print "\t\tTraining Data: TR Root Mean Square Error = {0:0.2f}".format(TR_rmse1)

    # Testing Accuracy
    TR_prediction = tree_reg.predict(strat_test_set)
    TR_mse = mean_squared_error(test_labels, TR_prediction)
    TR_rmse2.append(np.sqrt(TR_mse))
    print "\t\tTesting Data: TR Root Mean Square Error = {0:0.2f}".format(TR_rmse2[-1])

    #Linear Regression
    lin_reg = LinearRegression()
    lin_reg.fit(strat_train_set,train_labels.values.ravel())

    # Training Accuracy
    LR_prediction = lin_reg.predict(strat_train_set)
    LR_mse = mean_squared_error(train_labels, LR_prediction)
    LR_rmse1 = np.sqrt(LR_mse)
    print "\t\tTraining Data: LR Root Mean Square Error = {0:0.2f}".format(LR_rmse1)

    # Testing Accuracy
    LR_prediction = lin_reg.predict(strat_test_set)
    LR_mse = mean_squared_error(test_labels, LR_prediction)
    LR_rmse2.append(np.sqrt(LR_mse))
    print "\t\tTesting Data: LR Root Mean Square Error = {0:0.2f}".format(LR_rmse2[-1])

    # NN Model
    NN = MLPRegressor( activation='logistic', max_iter=5000, solver='lbfgs', hidden_layer_sizes=(3, 6), random_state=42)
    NN.fit(strat_train_set,train_labels.values.ravel())


    # Training Accuracy
    NN_prediction = NN.predict(strat_train_set)
    NN_mse = mean_squared_error(train_labels, NN_prediction)
    NN_rmse1 = np.sqrt(NN_mse)
    print "\t\tTraining Data: NN Root Mean Square Error = {0:0.2f}".format(NN_rmse1)

    # Testing Accuracy
    NN_prediction = NN.predict(strat_test_set)
    NN_mse = mean_squared_error(test_labels, NN_prediction)
    NN_rmse2.append(np.sqrt(NN_mse))
    print "\t\tTesting Data: NN Root Mean Square Error = {0:0.2f}".format(NN_rmse2[-1])

    #testing = [sum(rfr_rmse2), sum(lsvr_rmse2), sum(nusvr_rmse2), sum(svr_rmse2), sum(TR_rmse2), sum(LR_rmse2), sum(NN_rmse2)]
    #print [i[0] for i in sorted(enumerate(testing), key=lambda x: x[1])]
    #print ['{0:0.2f}'.format(i[1]) for i in sorted(enumerate(testing), key=lambda x: x[1])]
    #exit()
    """
    ####################################################
    ####################################################
    #The Best Regression Model is SVR
    # Decision Tree Regressor

    svr = svm.SVR(kernel='rbf', epsilon=0.01, C=10.0)
    svr.fit(strat_train_set, train_labels.values.ravel())

    # Training Accuracy
    svr_prediction = svr.predict(strat_train_set)
    svr_mse_train = mean_squared_error(train_labels, svr_prediction)
    svr_rmse_train = np.sqrt(svr_mse_train)
    print "\t\tTraining Data: SVR Root Mean Square Error = {0:0.2f}".format(svr_rmse_train)

    # Testing Accuracy
    svr_prediction = svr.predict(strat_test_set)
    svr_mse_test = mean_squared_error(test_labels, svr_prediction)
    svr_rmse_test = np.sqrt(svr_mse_test)
    print "\t\tTesting Data: SVR Root Mean Square Error = {0:0.2f}".format(svr_rmse_test)

    # Figures - Part 1
    fig1 = plt.figure(cluster_no)
    ax1 = fig1.add_subplot(111)


    ax1.plot(svr_prediction, c='b', label='Prediction (RMSE={0:0.2f})'.format(svr_rmse_test))
    ax1.plot(test_labels.values.tolist(), c='r', label='Actual')
    ax1.set_xlabel('Simulated Samples')
    ax1.set_ylabel('P(Success | Level, Robot Feedback, Prev. Score)')
    ax1.set_title('Success Rate using Cluster {0} Model on Test Data'.format(cluster_no))
    ax1.legend(bbox_to_anchor=(0.9, 0.82), loc=1, borderaxespad=0.,
               bbox_transform=plt.gcf().transFigure)
    #You have to create the folder 'fig' manually
    fig1.savefig('./fig/Performance on Test Data - Cluster_{0}.png'.format(cluster_no))



    #Train on all cluster_x Data and save to file
    svr_reg = svm.SVR(kernel='rbf', epsilon=0.01, C=10.0)
    svr_reg.fit(norm_features, labels.values.ravel())
    # You have to create the folder 'Output Models' manually
    with open('./Output Models/Cluster {} - SVR Model'.format(cluster_no),'w') as file:
        pickle.dump(svr_reg,file)

    print "Successfully saved Performance Model for cluster {}".format(cluster_no)


    #Data Simulation
    data_simulation = pd.DataFrame([[l, rf, ps] for l in [0, 1, 2, 3] for rf in [0, 1, 2] for ps in [1, -1, 2, -2, 3, -3, 4, -4]],columns=['Level', 'Robot_Feedback', 'Prev_Score'])
    norm_simulation = normalize_DataFrame(data_simulation.copy())

    prediction = svr_reg.predict(norm_simulation)

    #Figures - Part 2
    fig1 = plt.figure(cluster_no+3)
    ax1 = fig1.add_subplot(111)

    ax1.plot(range(0, len(prediction)), prediction, c='b', label= 'Prediction (RMSE={0:0.2f})'.format(svr_rmse_test))

    first_scatter = True
    for row in norm_simulation.itertuples():
        # row is Pandas Series in the form of [index, Level, Robot_Feedback, Prev_Score]
        label = actual_label(row, norm_features, labels)
        if label >= 0:
            if first_scatter:
                ax1.scatter(row[0], label, c='r' , label='Actual Values', marker='*', s=10)
                first_scatter = False
            else:
                ax1.scatter(row[0], label, c='r', marker='*', s=10)
    ax1.bar(left=[12, 36, 60, 84], height=[1.05,1.05,1.05,1.05], width=24, color=['k','w', 'k', 'w'], alpha=0.1 )
    ax1.set_xlabel('Simulated Samples')
    ax1.set_ylabel('P(Success | Level, Robot Feedback, Prev. Score)')
    ax1.set_title('Success Rate Prediction using Cluster {0} Model on Simulated Data'.format(cluster_no))
    ax1.axis([0, 96, -0.02, 1.05])
    ax1.text(7, 0.05, 'Level 1',style='italic', fontsize=10)
    ax1.text(31, 0.05, 'Level 2', style='italic', fontsize=10)
    ax1.text(55, 0.05, 'Level 3', style='italic', fontsize=10)
    ax1.text(79, 0.05, 'Level 4', style='italic', fontsize=10)
    ax1.legend(bbox_to_anchor=(0.9, 0.82), loc=1, borderaxespad=0.,
           bbox_transform=plt.gcf().transFigure)
    ax1.grid(alpha = 0.5, linestyle='-', linewidth=0.5)
    fig1.savefig('./fig/Performance on Simulated Data - Cluster_{0}.png'.format(cluster_no))
    #print "\tClose the figure in the background to continue ..."
    #plt.show()
    #plt.close()

print "\nEnd of the Program"

