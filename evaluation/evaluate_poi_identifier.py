#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
# score = clf.score(features_test, labels_test)
y_pred = clf.predict(features_test)
score = accuracy_score(labels_test, y_pred)
print(y_pred)
print("Num poi's in test set: %d" % y_pred.sum())
print("Total # of test set points: %d" % len(y_pred))
print("Confusion matrix......")
print(confusion_matrix(labels_test, y_pred))
print("Score: %f" % score)
print("Precision: %f; Recall: %f" % (precision_score(labels_test, y_pred), recall_score(labels_test, y_pred)))
print("F1 score: %f" % f1_score(labels_test, y_pred))



