#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
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
from sklearn import svm
from sklearn.metrics import accuracy_score
#clf = svm.LinearSVC()
clf = svm.SVC(C=10000.0, kernel="rbf")
t0 = time()
features_train = features_train[:len(features_train)//100]
labels_train = labels_train[:len(labels_train)//100]
clf.fit(features_train, labels_train)
print "Training time: ", round(time()-t0, 3), "s"
t0 = time()
y_pred = clf.predict(features_test)
print "Prediction time", round(time()-t0, 3), "s"
print "Accuracy = %f" % accuracy_score(y_pred, labels_test)
print "Element 10: %d" % y_pred[10]
print "Element 26: %d" % y_pred[26]
print "Element 50: %d" % y_pred[50]
print "Total Chris: %d" % y_pred.sum()
print features_train.shape

#########################################################


