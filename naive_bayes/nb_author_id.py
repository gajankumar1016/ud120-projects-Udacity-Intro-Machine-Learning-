#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
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
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
gnb = GaussianNB()
t0 = time()
clf = gnb.fit(features_train, labels_train)
print "Training time: ", round(time()-t0, 3), "s"
t0 = time()
y_pred = clf.predict(features_test)
print "prediction time: ", round(time()-t0, 3), "s"
total_pts = features_test.shape[0] #equivalent to len(labels_test)
mislabeled_pts = (labels_test != y_pred).sum()
print("Number of mislabeled points out of a total %d points : %d"
	% (total_pts, mislabeled_pts))
#print("Accuracy: %f" % (1 - mislabeled_pts / float(total_pts)))
#print("clf.score() -> accuracy = %f" % clf.score(features_test, labels_test))
print("Using sklearn.metrics: Accuracy = %f" % accuracy_score(y_pred, labels_test))


#########################################################


