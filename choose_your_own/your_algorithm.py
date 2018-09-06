#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
# plt.xlim(0.0, 1.0)
# plt.ylim(0.0, 1.0)
# plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
# plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
# plt.legend()
# plt.xlabel("bumpiness")
# plt.ylabel("grade")
# plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.metrics import accuracy_score
from time import time

def fit_and_predict(clf):
	t0 = time()
	clf.fit(features_train, labels_train)
	print "Training time: %f" % round(time()-t0,3)
	t0 = time()
	y_pred = clf.predict(features_test)
	print "Prediction time: %f" % round(time()-t0,3)
	print "Accuracy: %f" % accuracy_score(y_pred, labels_test)

##NaiveBayes
print "\nUsing GaussianNB..."
clf = GaussianNB()
fit_and_predict(clf)

##SVM
print "\nUsing SVM..."
clf = svm.SVC(C=10000.0, kernel="rbf")
fit_and_predict(clf)

##Decision Tree
print "\nUsing DecisionTreeClassifier..."
clf = tree.DecisionTreeClassifier(min_samples_split = 22)
fit_and_predict(clf)

##KNN classifier
print "\nUsing KNeighborsClassifier..."
clf = KNeighborsClassifier(n_neighbors=7)
fit_and_predict(clf)

##AdaBoostClassifier
print "\nUsing AdaBoostClassifier..."
clf = AdaBoostClassifier()
fit_and_predict(clf)

##RandomForestClassifier
print "\nUsing RandomForestClassifier..."
clf = RandomForestClassifier(n_estimators=50)
fit_and_predict(clf)


try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
