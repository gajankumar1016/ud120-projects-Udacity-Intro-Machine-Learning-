#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn import tree
from sklearn.metrics import accuracy_score

# features_train = features_train[:len(features_train)//100]
# labels_train = features_train[:len(features_train)//100]

clf = tree.DecisionTreeClassifier(min_samples_split=40)
t0 = time()
clf.fit(features_train, labels_train)
print "Training time: %f" % round(time()-t0, 3)
t0 = time()
y_pred = clf.predict(features_test)
print "Prediction time: %f" % round(time()-t0, 3)
print "Accuracy: %f" % accuracy_score(y_pred, labels_test)
shape = features_train.shape
print "Shape: {0}".format(shape)

#########################################################


