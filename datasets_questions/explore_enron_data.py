#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

def print_keys(dict):
	for key in dict.keys():
		print key

def print_values(dict):
	for value in dict.values():
		print value

def print_dict(dict):
	for k, v in dict.iteritems():
		print k, v

def find_num_poi(dict):
	count = 0
	for v in dict.values():
		#same as if v["poi"] == 1
		if v["poi"]:
			count += 1
	return count

import random
def print_features(dict):
	rand_dict = random.choice(dict.values())
	for k in rand_dict.keys():
		print k


# print enron_data['LAY KENNETH L']['total_payments']
# print enron_data['SKILLING JEFFREY K']['total_payments']
# print enron_data['FASTOW ANDREW S']['total_payments']

total = 0
count_salary = 0
count_email = 0
for v_dict in enron_data.values():
	total += 1
	for k, v in v_dict.items():
		if k == 'salary':
			if v != 'NaN':
				count_salary += 1
		if k == 'email_address':
			if v != 'NaN':
				count_email += 1

print "Num quantified salary: %d" % count_salary
print "Num known email: {0}".format(count_email)
print "Total persons in dataset: %d" % total



