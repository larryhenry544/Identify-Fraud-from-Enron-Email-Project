#!/usr/bin/python2

import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot
import warnings
import pprint
from time import time

warnings.filterwarnings("ignore")
sys.path.append("../tools/")

## importing the sklearn functions
from sklearn import preprocessing, tree
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, RobustScaler, Binarizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier


# Create function to display data that uses the featureFormat script
def plotData(data_set, first_feature, second_feature):
    """
    Function takes a dict, 2 strings, and shows a 2d plot of 2 features
    """
    data = featureFormat(data_set, [first_feature, second_feature])
    for point in data:
        x = point[0]
        y = point[1]
        matplotlib.pyplot.scatter(x, y)

    matplotlib.pyplot.xlabel(first_feature)
    matplotlib.pyplot.ylabel(second_feature)
    matplotlib.pyplot.show()


#######################################################################################################
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

#features_list = ['poi','salary'] # You will need to use more features

# all units are in US dollars
financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                      'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                      'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']

# units are generally number of emails messages; notable exception is 'email_address', which is a text string
email_features = ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
                  'shared_receipt_with_poi']

# boolean, represented as integer
poi_label = ['poi']

fraction_features = ['fraction_to_this_person_from_poi', 'fraction_form_this_person_to_poi']

all_features_list = poi_label + financial_features + email_features + fraction_features
all_features_list.remove('email_address')

my_features = ['total_stock_value', 'restricted_stock', 'exercised_stock_options', 'salary', 'bonus', 'deferred_income',
               'long_term_incentive']

features_list = poi_label + my_features + fraction_features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# data_dict = pd.read_pickle("final_project_dataset.pkl")

print "Data points: ", pprint.pprint(data_dict)
print "Total number of data points: ", float(len(data_dict))
print "Total number of data points: ", type(data_dict), "\n"

# Allocation across classes (POI/non-POI)
poi = 0
for person in data_dict:
    if data_dict[person]['poi'] == True:
        poi += 1
print "Total number of poi:", poi
print "Total number of non-poi:", (len(data_dict) - poi)

# Number of features used
all_features = data_dict[data_dict.keys()[0]].keys()
print "There are", float(len(all_features)), "features for each person in the dataset, and",  float(len(features_list)),\
    "features are used"



#######################################################################################################
### Task 2: Remove outliers

# Are there features with many missing values? etc.
missing_values = dict()
for feature in all_features:
    missing_values[feature] = 0
for person in data_dict:
    for feature in data_dict[person]:
        if data_dict[person][feature] == "NaN":
            missing_values[feature] += 1

print "The number of missing values for each feature: "
for feature in missing_values:
    print feature, ": ",  missing_values[feature]
print "\n"

identity = []
for person in data_dict:
    if data_dict[person]['total_payments'] != "NaN":
        identity.append((person, data_dict[person]['total_payments']))
print "Outlier:"
print (sorted(identity, key = lambda x: x[1], reverse=True)[0:4])

# remove persons who don't have any feature value
for name in data_dict:
    data_without_feature = True
    for feature in data_dict.get(name):
        if feature != 'poi' and data_dict.get(name)[feature] != 'NaN':
            data_without_feature = False
            break
    if data_without_feature==True:
        print 'Outliers name = ', name
print "\n"

# Find persons whose financial features are all "NaN"
fi_nan_dict = {}
for person in data_dict:
    fi_nan_dict[person] = 0
    for feature in financial_features:
        if data_dict[person][feature] == "NaN":
            fi_nan_dict[person] += 1
sorted(fi_nan_dict.items(), key=lambda x: x[1])

# Find persons whose email features are all "NaN"
email_nan_dict = {}
for person in data_dict:
    email_nan_dict[person] = 0
    for feature in email_features:
        if data_dict[person][feature] == "NaN":
            email_nan_dict[person] += 1
sorted(email_nan_dict.items(), key=lambda x: x[1])

# Visualize data to identify outliers
# plotData(data_dict, 'total_payments', 'total_stock_value')
# plotData(data_dict, 'from_poi_to_this_person', 'from_this_person_to_poi')
# plotData(data_dict, 'salary', 'bonus')
# plotData(data_dict, 'total_payments', 'other')

# Remove outliers
data_dict.pop("TOTAL", 0)
data_dict.pop("LOCKHART EUGENE E", 0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)

# Visualize data after removing outliers
# plotData(data_dict, 'total_stock_value', 'total_payments')
# plotData(data_dict, 'from_poi_to_this_person', 'from_this_person_to_poi')
# plotData(data_dict, 'salary', 'bonus')
# plotData(data_dict, 'restricted_stock', 'exercised_stock_options')
# plotData(data_dict, 'long_term_incentive', 'deferred_income')



#######################################################################################################
### Task 3: Create new feature(s)

# check how many people have email data
has_email_data = 0
for name in data_dict:
    if (
            data_dict.get(name)['to_messages'] != 'NaN' and
            data_dict.get(name)['from_poi_to_this_person'] != 'NaN' and
            data_dict.get(name)['from_messages'] != 'NaN' and
            data_dict.get(name)['from_this_person_to_poi'] != 'NaN' and
            data_dict.get(name)['shared_receipt_with_poi'] != 'NaN' and
            data_dict.get(name)['email_address'] != 'NaN'
    ):
        has_email_data = has_email_data + 1

print 'Percentage of persons who has email data = ', (float(has_email_data) / float(len(data_dict))) * 100

## Creating new features
# create new features fraction_to_this_person_from_poi and fraction_form_this_person_to_poi
# since everyone has to have email data, just data is not available to us, Assigning 0.1 as default value for persons
# who don't have email data.
for name in data_dict:
    if (data_dict.get(name)['to_messages'] != 'NaN' and data_dict.get(name)['from_poi_to_this_person'] != 'NaN' and
            data_dict.get(name)['from_messages'] != 'NaN' and data_dict.get(name)['from_this_person_to_poi'] != 'NaN'):
        data_dict.get(name)['fraction_to_this_person_from_poi'] = float(
            data_dict.get(name)['from_poi_to_this_person']) / float(data_dict.get(name)['from_messages'])
        data_dict.get(name)['fraction_form_this_person_to_poi'] = float(
            data_dict.get(name)['from_this_person_to_poi']) / float(data_dict.get(name)['to_messages'])
    else:
        data_dict.get(name)['fraction_to_this_person_from_poi'] = 0.1
        data_dict.get(name)['fraction_form_this_person_to_poi'] = 0.1

# count NaN values for every feature to see what feature are most reliable
feature_analysis = dict()
for feature in all_features_list:
    for name in data_dict:
        if feature_analysis.get(feature, None) == None:
            feature_analysis[feature] = 0
        if data_dict.get(name)[feature] == 'NaN':
            feature_analysis[feature] = feature_analysis.get(feature) + 1

# checkout percentage of existing features data (loan_advances  has only 2.0979020979 % of data)
feature_analysis = sorted(feature_analysis.items(), key=lambda x: x[1], reverse=False)
for feature in feature_analysis:
    print feature[0], ' = ', ((float(len(data_dict)) - float(feature[1])) / float(len(data_dict))) * 100, '%'

# fill missing feature values
for name in data_dict:
    for feature in data_dict[name]:
        if data_dict[name][feature] == 'NaN':
            data_dict[name][feature] = 0

### Store to my_dataset for easy export below.
my_dataset = data_dict
# print "my_dataset:", pprint.pprint(my_dataset)

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

scaler = preprocessing.MinMaxScaler()
print "scaler:", scaler
features = scaler.fit_transform(features)
print "features", features

# intelligently select features (univariate feature selection)
selector = SelectKBest(chi2, k='all')
selector.fit(features, labels)
scores = zip(features_list[1:], selector.scores_)
scores_list = sorted(scores, key=lambda x: x[1], reverse=True)
print "scores_list:", pprint.pprint(scores_list)

# Find the best number of features
n_features = np.arange(1, len(features_list))
print "n_features:", n_features

## DecisionTreeClassifier
# Create a pipeline with feature selection and classification
pipe = Pipeline([
    ('select_features', SelectKBest(chi2)),
    ('classify', tree.DecisionTreeClassifier())
])

param_grid = [
    {
        'select_features__k': n_features
    }
]

# Use GridSearchCV to automate the process of finding the optimal number of features
# tree_clf = GridSearchCV(pipe, param_grid=param_grid, scoring='f1', cv=5, iid=False)
tree_clf = GridSearchCV(pipe, param_grid=param_grid, scoring='f1', cv=5)
tree_clf.fit(features, labels)

print "tree_clf.best_params_", tree_clf.best_params_

##RandomForestClassifier
# Create a pipeline with feature selection and classification
pipe = Pipeline([
    ('select_features', SelectKBest(chi2)),
    ('classify', RandomForestClassifier(max_depth=None, min_samples_split=2))
])

param_grid = [
    {
        'select_features__k': n_features
    }
]

# Use GridSearchCV to automate the process of finding the optimal number of features
# forest_clf = GridSearchCV(pipe, param_grid=param_grid, scoring='f1', cv=5, iid=False)
forest_clf = GridSearchCV(pipe, param_grid=param_grid, scoring='f1', cv=5)
forest_clf.fit(features, labels)

print "forest_clf.best_params_", forest_clf.best_params_


#######################################################################################################
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# clf = Pipeline([
#     ('scaling', preprocessing.MinMaxScaler()),
#     ('pca', decomposition.PCA(n_components=15, svd_solver='full')),
#     ('select_best', SelectKBest(k=10)),
#     ('algorithm', GaussianNB())
# ])

# Use the 9 first features of the KBest selector
features_list = ['poi'] + [score[0] for score in scores_list][:9]

# Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)


def cross_validate(clf, name):
    """
    Receives a classifier and the name of the classifier and performs cross_validation scoring on the features and
    labels in the global scope. Returns a pandas DataFrame with the results
    """

    accuracy_scores = cross_val_score( clf, features, labels, cv=5, scoring='accuracy')

    precision_scores = cross_val_score(clf, features, labels, cv=5, scoring='precision')

    recall_scores = cross_val_score(clf, features, labels, cv=5, scoring='recall')

    f1_scores = cross_val_score(clf, features, labels, cv=5, scoring='f1')

    accuracy = '%0.2f (+/- %0.2f)' % (accuracy_scores.mean(), accuracy_scores.std() * 2)

    precision = '%0.2f (+/- %0.2f)' % (precision_scores.mean(), precision_scores.std() * 2)

    recall = '%0.2f (+/- %0.2f)' % (recall_scores.mean(), recall_scores.std() * 2)

    f1 = '%0.2f (+/- %0.2f)' % (f1_scores.mean(), recall_scores.std() * 2)

    return pd.DataFrame(index=[name],
                        data={'Accuracy': [accuracy],
                              'Precision': [precision],
                              'Recall': [recall],
                              'F1': [f1]
                              }
                        )

df = pd.DataFrame()
n_features = len(features_list) - 1

clf1 = GaussianNB()
df1 = cross_validate(clf1, 'GaussianNB')


clf2 = tree.DecisionTreeClassifier()
df2 = cross_validate(clf2, 'DecisionTreeClassifier')
df = df1.append(df2)


clf3 = RandomForestClassifier(n_estimators=10,
                              max_features=n_features,
                              max_depth=None,
                              min_samples_split=2)
df3 = cross_validate(clf3, 'RandomForestClassifier')
df = df.append(df3)


# sklearn.linear_model.LogisticRegression
clf4 = LogisticRegression(C=1e5)
df4 = cross_validate(clf4, 'LogisticRegression')
df = df.append(df4)


# KMeans
clf5 = KMeans(n_clusters=2, random_state=0)
df5 = cross_validate(clf5, 'KMeans')
df = df.append(df5)
print "df5:", pprint.pprint(df)



#######################################################################################################
### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
# from sklearn.model_selection import train_test_split
# features_train, features_test, labels_train, labels_test = \
#     train_test_split(features, labels, test_size=0.3, random_state=42)

clf = tree.DecisionTreeClassifier()
# Define the configuration of parameters to test with the
# Decision Tree Classifier
param_grid = {'criterion': ['gini', 'entropy'],
              'min_samples_split': [2, 4, 6, 8, 10, 20],
              'max_depth': [None, 5, 10, 15, 20],
              'max_features': [None, 'sqrt', 'log2', 'auto']}

# Use GridSearchCV to find the optimal hyperparameters for the classifier
# tree_clf = GridSearchCV(clf, param_grid=param_grid, scoring='f1', cv=5, iid=False)
tree_clf = GridSearchCV(clf, param_grid=param_grid, scoring='f1', cv=5)
tree_clf.fit(features, labels)

# Get the best algorithm hyperparameters for the Decision Tree
print "tree_clf.best_params_:", tree_clf.best_params_


# A Dataframe to use in the reporting
df_final = cross_validate(tree_clf.best_estimator_, 'Tuned DecisionTreeClassifier')


# Tune the parameters for RandomForestClassifier
clf3 = RandomForestClassifier(max_depth=None, min_samples_split=2)

param_grid = {
    "n_estimators": [9, 18, 27, 36],
    "max_depth": [None, 1, 5, 10, 15],
    "min_samples_leaf": [1, 2, 4, 6]}

# Use GridSearchCV to find the optimal hyperparameters for the classifier
# forest_clf = GridSearchCV(clf3, param_grid=param_grid, scoring='f1', cv=5, iid=False)
forest_clf = GridSearchCV(clf3, param_grid=param_grid, scoring='f1', cv=5)
forest_clf.fit(features, labels)
# Get the best algorithm hyperparameters for the Decision Tree
print "forest_clf.best_params_:", forest_clf.best_params_

df_final = df_final.append(cross_validate(forest_clf.best_estimator_, 'Tuned RandomForestClassifier'))
print "df_final:", pprint.pprint(df_final)

# Use the DecisionTreeClassifier for tester.py
clf = tree_clf.best_estimator_


#######################################################################################################
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

test_classifier(clf, my_dataset, features_list)
# print clf.named_steps['select_best'].scores_

dump_classifier_and_data(clf, my_dataset, features_list)
