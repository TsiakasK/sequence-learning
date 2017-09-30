## Reading the raw Data
- This code reads the raw data based on the order specified in /Data Analysis/clusters
- There is a dictionary for each cluster where the keys are tuples of (Level, Robot_Feedback, Prev_score) and the values are the calculted success rate
- The success rate =  #wins/(#wins+#losses)
- All these dictionaries are then saved into a list


## Regression Models
- Each cluster data are read and saved as Pandas DataFrame and they are split into training (80%) and testing (20%) data
- The features DataFrame is normalized by a custom function, normalize_DataFrame() located on the top of the file.
- The data are stratified to make sure the percentage of each level is similar in the training set, testing set, and the cluster set (training + testing sets). For example, if 40% of the cluster data belong to Level 0, then ~40% of the data in the training set will also belong to Level 0.
- 7 regression models were tested on that data, which are Random Forst, LinearSVR, NuSVR, SVR, Decision Tree, Linear Regression, and Neural Notwork. The code for that is commented in the middle of the .py file. and at the end of the  file, there is a commented code that was used to determine the parameters for SVR and NN.
- The order of the models based on the performance is:
	1. Decision Tree and SVR with RBF kernel
	2. Linear Regression
	3. NuSVR and LinearSVR
	4. Random Forest
	5. Neural Network (Good Accuracy on the training set, but very bad on the test set)
- The Decision Tree model was chosen as the best model and it was fit using the training data.
- Then, it was used to plot the success rate prediction on the test data and calculte the RMSE. The plots are saved in /fig
- After that, the model was fit into the cluster data (training and testing data) to predict the performance of simulted data. The new models are saved in a folder called /Outpus Models/ in order to be used later in RL
- Since there are 4 levels, 3 Robot Feedbacks, and 8 possible previous scores, the length of the simulated data is 96 (4x3x8).
- At the end, the predicted simulated data vs actual data are plotted and saved in /fig

NOTE: This is first draft of the README.md, and it might be updated later.