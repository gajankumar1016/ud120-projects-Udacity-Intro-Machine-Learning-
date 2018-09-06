#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
#Remove point corresponding to total
data_dict.pop("TOTAL", 0)
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below
for point in data:
	salary = point[0]
	bonus = point[1]
	matplotlib.pyplot.scatter(salary, bonus)

for point in data:
	salary = point[0]
	bonus = point[1]
	if (salary > 1e6 and bonus > 5e6):
		print "Salary: " + str(salary)
		print "Bonus: " + str(bonus)

for k, v_dict in data_dict.iteritems():
	if v_dict["salary"] > 1e6 and v_dict["bonus"] > 5e6 and v_dict["poi"]:
		print k




matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()



