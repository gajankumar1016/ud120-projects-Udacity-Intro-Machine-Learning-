#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# Possible Features: {'salary': 365788, 'to_messages': 807, 'deferral_payments': 'NaN', 'total_payments': 1061827, 
# 'exercised_stock_options': 'NaN', 'bonus': 600000, 'restricted_stock': 585062, 'shared_receipt_with_poi': 702, 
# 'restricted_stock_deferred': 'NaN', 'total_stock_value': 585062, 'expenses': 94299, 'loan_advances': 'NaN', 
# 'from_messages': 29, 'other': 1740, 'from_this_person_to_poi': 1, 'poi': False, 'director_fees': 'NaN', 'deferred_income': 'NaN', 
# 'long_term_incentive': 'NaN', 'email_address': 'mark.metts@enron.com', 'from_poi_to_this_person': 38}

features_list = ['poi','salary', 'bonus', 'total_stock_value', 'expenses', 'from_this_person_to_poi'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
import sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, make_scorer
clfs_dict = {
'GaussianNB': {'clf': GaussianNB()}, 
'DecisionTreeClassifier': {'clf': DecisionTreeClassifier(), 'parameters': {'min_samples_split': [3, 5, 10, 15, 20, 25, 30]}}, 
'svm': {'clf': svm.SVC(), 'parameters': {'kernel': ['rbf'], 'C': [1, 5, 10, 100]}},
'KNeighborsClassifier': {'clf': KNeighborsClassifier(), 'parameters': {'n_neighbors':[5, 7, 9]}}, # looks the best so far F1 = 0.57
'AdaBoostClassifier': {'clf': AdaBoostClassifier()},
'RandomForestClassifier': {'clf': RandomForestClassifier(), 'parameters': {'n_estimators':[25, 35]}}
}
clf_name = 'RandomForestClassifier'
clf_untuned = clfs_dict[clf_name]['clf']
clf = GridSearchCV(clf_untuned, clfs_dict[clf_name]['parameters'], scoring=make_scorer(f1_score)) # scoring=make_scorer(f1_score)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# features = MinMaxScaler().fit_transform(features)
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

clf.fit(features_train, labels_train)
print(clf.best_estimator_)
clf = clf.best_estimator_

y_pred = clf.predict(features_test)
score = accuracy_score(labels_test, y_pred)
print("Num poi's in test set: %d" % y_pred.sum())
print("Total # of test set points: %d" % len(y_pred))
print("Confusion matrix......")
print(confusion_matrix(labels_test, y_pred))
print("Score: %f" % score)
print("Precision: %f; Recall: %f" % (precision_score(labels_test, y_pred), recall_score(labels_test, y_pred)))
print("F1 score: %f" % f1_score(labels_test, y_pred))

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)