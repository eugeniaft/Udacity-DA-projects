# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 10:36:53 2016

@author: Eugenia
"""

#!/usr/bin/python

import sys
import pickle
import numpy as np
sys.path.append("../tools/")
import matplotlib.pyplot as plt
from feature_format import featureFormat, targetFeatureSplit
from sklearn import preprocessing, tree, linear_model
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tester import test_classifier
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'bonus', 'long_term_incentive', 'deferred_income',
                 'deferral_payments', 'to_messages', 'other', 'director_fees',
                 'expenses', 'total_payments', 'exercised_stock_options',
                 'restricted_stock', 'total_stock_value', 'loan_advances',
                 'from_messages', 'shared_receipt_with_poi',
                 'restricted_stock_deferred']
                     
print features_list                 

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

#Exploring dataset
print "SKILLING JEFFREY K", data_dict["SKILLING JEFFREY K"]
print "LAY KENNETH L", data_dict["LAY KENNETH L"]
print "FASTOW ANDREW S", data_dict["FASTOW ANDREW S"]

#how many people are in the data?
total = len(data_dict)
print "Number of people in dataset: %f" %total

#How many POIs?
count = 0
for i in data_dict:
    if data_dict[i]['poi'] == True:
        count +=1
print "Number of POIs: %f" % count

#For each person, how many features are available?
for i in data_dict:
    feats = len(data_dict[i])
print "Number of features: %f" % feats

#For each feature, count NaNs and number of POIs with NaNs    
NaNs_pois = {}

for k,v in data_dict.iteritems(): 
    for i in v:
        if i != 'poi':
            NaNs_pois[i]={'NaN': 0, 'poi': 0}
for k,v in data_dict.iteritems():
    for i in v:
        if str(v[i]).lower() == 'nan':
            NaNs_pois[i]['NaN']+=1
            if v['poi'] == True:
                NaNs_pois[i]['poi']+=1

print NaNs_pois

#remove features whith lots of missing values (>102) and where
#more than 70% (>12) of POIs have missing values
features_list.remove('loan_advances')
features_list.remove('restricted_stock_deferred')
features_list.remove('director_fees')
features_list.remove('deferral_payments')

print features_list

### Task 2: Remove outliers
print data_dict.keys()

data_dict.pop('TOTAL')
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')

#All features have missing values
data_dict.pop('LOCKHART EUGENE E')

### Task 3: Create new feature(s)
#Adding ratio of sent messages to received messages 
#Adding ratio of POI messages to total messages
            
for key in data_dict:
    try:
        scatter = plt.scatter(data_dict[key]['from_messages'], data_dict[key]['to_messages'])
    except:
        print "Error"
        
for key in data_dict:
    #new feature
    try:
        sent_recieved_ratio = float(data_dict[key]['from_messages'])/ data_dict[key]['to_messages']
        data_dict[key]['sent_received_ratio'] = sent_recieved_ratio         
    except:
        data_dict[key]['sent_received_ratio'] = 'NaN'                 

print "SKILLING JEFFREY K", data_dict["SKILLING JEFFREY K"]['sent_received_ratio']             

#Adding new feature to list of features
features_list.append('sent_received_ratio')
print len(features_list)

### Store to my_dataset for easy export below.
my_dataset = data_dict
                 
### Extract features and labels from dataset for local testing
#Funtion converts dictionary to numpy array of features
data = featureFormat(my_dataset, features_list, sort_keys = True)

#Given a numpy array this function separates out the first feature
#and put it into its own list, this is the what you want to predict, 
#in our case first feature is 'poi'
labels, features = targetFeatureSplit(data)

labels = np.array(labels)
features = np.array(features)

#scaling features
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)

#Splitting data into test and training sets
sss = StratifiedShuffleSplit(labels, n_iter=10)
     
# Univariate feature selection
# Let's look at the scores from SelectKBest for all features
selector=SelectKBest(f_classif, k='all')
selector.fit_transform(features,labels) 
feature_scores = ['%.2f' % elem for elem in selector.scores_ ]
feature_scores_pvalues = ['%.3f' % elem for elem in  selector.pvalues_ ]
features_selected_tuple=[(features_list[i+1], feature_scores[i], feature_scores_pvalues[i]) for i in selector.get_support(indices=True)]
features_selected_tuple = sorted(features_selected_tuple, key=lambda feature: float(feature[1]) , reverse=True)
print ' '
print 'Selected Features, Scores, P-Values'
print features_selected_tuple
     
### Task 4: Try a varity of classifiers

clf_GNB = GaussianNB()
clf_GNB.fit(features, labels)
scores_rec_GNB = cross_val_score(clf_GNB, features, labels,cv=sss,scoring='f1')
print "GaussianNB", scores_rec_GNB

clf_DT = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=1)
clf_DT.fit(features, labels)
scores_rec_DT = cross_val_score(clf_DT, features, labels, cv=sss, scoring='f1')
print "DecisionTree", scores_rec_DT

clf_RF = RandomForestClassifier(criterion='entropy', min_samples_split=1, 
                                n_estimators=10)
clf_RF.fit(features, labels)
scores_rec_RF = cross_val_score(clf_RF, features, labels, cv=sss,scoring='f1')
print "RandomForest", scores_rec_RF

clf_AB = AdaBoostClassifier(n_estimators=10)
clf_AB.fit(features, labels)
scores_rec_AB = cross_val_score(clf_AB, features, labels,cv=sss,scoring='f1')
print "AdaBoost", scores_rec_AB

clf_LR = linear_model.LogisticRegression(C=1000)
clf_LR.fit(features, labels)
scores_rec_LR = cross_val_score(clf_LR, features, labels,cv=sss,scoring='f1')
print "LogisticRegression", scores_rec_LR

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
##Parameter tuning only GNB, DT, and LG

#GaussianNaiveBayes
clf_GNB = GaussianNB()
pca = PCA()
skb = SelectKBest(f_classif)

pipe_GNB= Pipeline(steps=[('SKB', skb), ('pca', pca), ('GNB', clf_GNB)])

params_grid = {"pca__n_components":range(1,5), "pca__whiten": [True, False]}
SKM_params = {"SKB__k":range(5,15)}

params_grid.update(SKM_params)

gridGNB=GridSearchCV(pipe_GNB, params_grid, scoring='recall', cv=sss)

gridGNB.fit(features, labels)

print "GNB Optimal Parameters:", gridGNB.best_params_

features_k= gridGNB.best_params_['SKB__k']
SKB_k=SelectKBest(f_classif, k=features_k)
SKB_k.fit_transform(features,labels) 
feature_scores = SKB_k.scores_
features_selected=[features_list[i+1]for i in SKB_k.get_support(indices=True)]
features_scores_selected=[feature_scores[i]for i in SKB_k.get_support(indices=True)]
print ' '
print 'Selected Features', features_selected
print 'Feature Scores', features_scores_selected
        
print gridGNB.best_score_ 

#Decision Tree

clf_DT_tun = tree.DecisionTreeClassifier()
pca = PCA()
skb = SelectKBest(f_classif)

pipeDT = Pipeline(steps=[('SKB', skb), ('pca', pca), ('DT', clf_DT_tun)])
                    
params_grid = {"pca__n_components":range(1,5), "pca__whiten": [True, False]}
DT_params = {"DT__criterion":['entropy', 'gini'],"DT__min_samples_split": [1,2,3,4,5],
             "DT__max_features":['sqrt', 'log2', None]}
SKM_params = {"SKB__k":range(5,15)}
params_grid.update(DT_params)
params_grid.update(SKM_params)

gridDT =  GridSearchCV(pipeDT, params_grid, scoring='recall', cv=sss)
gridDT.fit(features, labels)
print "DT Optimal Parameters:", gridDT.best_params_

features_k= gridDT.best_params_['SKB__k']
SKB_k=SelectKBest(f_classif, k=features_k)
SKB_k.fit_transform(features,labels) 
feature_scores = SKB_k.scores_
features_selected=[features_list[i+1]for i in SKB_k.get_support(indices=True)]
features_scores_selected=[feature_scores[i]for i in SKB_k.get_support(indices=True)]
print ' '
print 'Selected Features', features_selected
print 'Feature Scores', features_scores_selected
        
print gridDT.best_score_ 

#Logistic Regression

clf_logistic = linear_model.LogisticRegression(class_weight = 'auto', 
                                              random_state=245)
pca = PCA()
skb = SelectKBest(f_classif)
pipe_LR= Pipeline(steps=[('SKB', skb), ('pca', pca), ('LR', clf_logistic)])

Cs = np.logspace(-4, 4, 3)
tols = [10**-1, 10**-5, 10**-10]                                 

params_grid = {"pca__n_components":range(1,5), "pca__whiten": [True, False]}
SKM_params = {"SKB__k":range(5,15)}
LR_params = {"LR__C":Cs, "LR__tol": tols} 
params_grid.update(SKM_params)
params_grid.update(LR_params)

gridLR = GridSearchCV(pipe_LR, params_grid, scoring='recall', cv=sss) 
gridLR.fit(features, labels)               
print "LR Optimal Parameters:", gridLR.best_params_

features_k= gridLR.best_params_['SKB__k']
SKB_k=SelectKBest(f_classif, k=features_k)
SKB_k.fit_transform(features,labels) 
feature_scores = SKB_k.scores_
features_selected=[features_list[i+1]for i in SKB_k.get_support(indices=True)]
features_scores_selected=[feature_scores[i]for i in SKB_k.get_support(indices=True)]
print ' '
print 'Selected Features', features_selected
print 'Feature Scores', features_scores_selected
        
print gridLR.best_score_
 
##Testing algorithms using tester function and optimal parameters

###Gaussian Naive Bayes
clf_GNB_opt = GaussianNB()
pca = PCA(n_components=3, whiten=True)
scaler = preprocessing.MinMaxScaler()

features_GNB = ['poi', 'salary', 'bonus', 'long_term_incentive', 'deferred_income', 
                'total_payments', 'exercised_stock_options', 'restricted_stock',
                'total_stock_value', 'shared_receipt_with_poi']

pipeGNB_opt= Pipeline(steps=[('scaler', scaler ), ('pca', pca), ('GNB', clf_GNB_opt)])

print test_classifier(pipeGNB_opt, my_dataset, features_GNB, folds=1000)

n_comps = [1, 2, 4, 5]

#Testing performance with different number of compponents for pca
for i in n_comps:
    pca = PCA(n_components=i, whiten=True)
    pipeGNB_opt= Pipeline(steps=[('scaler', scaler ), ('pca', pca), ('GNB', clf_GNB_opt)])
    print test_classifier(pipeGNB_opt, my_dataset, features_GNB, folds=1000)

##Decision Tree optimal
clf_DT_opt = tree.DecisionTreeClassifier(criterion='entropy',
                                         min_samples_split=4)
pca = PCA(n_components=4, whiten=True)

features_DT=['poi', 'salary', 'bonus', 'long_term_incentive', 'deferred_income', 
              'to_messages', 'other', 'expenses', 'total_payments', 
              'exercised_stock_options', 'restricted_stock',
              'total_stock_value', 'shared_receipt_with_poi']

pipeDT_opt= Pipeline(steps=[('scaler', scaler), ('pca', pca), ('DT', clf_DT_opt)])

print test_classifier(pipeDT_opt, my_dataset, features_DT, folds=1000)  

n_comps = [1, 2, 3, 5]

#Testing performance with different number of compponents for pca
for i in n_comps:
    pca = PCA(n_components=i, whiten=True)
    pipeDT_opt= Pipeline(steps=[('scaler', scaler ), ('pca', pca), ('DT', clf_DT_opt)])
    print test_classifier(pipeDT_opt, my_dataset, features_DT, folds=1000)


##Logistic Regression
clf_LR_opt = linear_model.LogisticRegression(class_weight = 'auto',
                                             tol= 0.1,
                                             C= 0.0001)
pca = PCA(n_components=1)

features_LR=['poi', 'salary', 'bonus', 'long_term_incentive', 'deferred_income', 
             'to_messages', 'other', 'expenses', 'total_payments', 
             'exercised_stock_options', 'restricted_stock', 
             'total_stock_value', 'shared_receipt_with_poi']

pipeLR_opt= Pipeline(steps=[('scaler', scaler), ('pca', pca), ('GNB', clf_LR_opt)])

pipeLR_opt.fit(features, labels)

print test_classifier(pipeLR_opt, my_dataset, features_LR, folds=1000)  

n_comps = [2, 3, 4, 5]

#Testing performance with different number of compponents for pca
for i in n_comps:
    pca = PCA(n_components=i, whiten=True)
    pipeLR_opt= Pipeline(steps=[('scaler', scaler ), ('pca', pca), ('LR', clf_LR_opt)])
    print test_classifier(pipeLR_opt, my_dataset, features_LR, folds=1000)

#Other Evaluation Techniques
#Split data into training and test sets

def score_res(clf, labels, predictions):
    print clf + " " + "Accuracy Score: %f" % accuracy_score(labels_test, predictions)
    print clf + " " + "Precision Score: %f" % precision_score(labels_test, predictions)
    print clf + " " + "Recall Score: %f" % recall_score(labels_test, predictions)
 
#GaussianNB   
data_GNB = featureFormat(my_dataset, features_GNB, sort_keys = True)
labels, features = targetFeatureSplit(data_GNB)
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.3, random_state = 42)                                        

pipeGNB_opt.fit(features_train, labels_train)
preds=pipeGNB_opt.predict(features_test)
 
score_res("GaussianNB", labels_test, preds) 

#Logistic Regression 
data_LR = featureFormat(my_dataset, features_LR, sort_keys = True)
labels, features = targetFeatureSplit(data_LR)
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.3, random_state = 42)                                        

pipeLR_opt.fit(features_train, labels_train)
preds=pipeLR_opt.predict(features_test)
 
score_res("LogisiticRegression", labels_test, preds) 

#Chosen algorithm: Gaussian Naive Bayes
clf_GNB_opt = GaussianNB()
pca = PCA(n_components=3, whiten=True)
scaler = preprocessing.MinMaxScaler()

features_list=['poi', 'salary', 'bonus', 'long_term_incentive', 'deferred_income', 
              'total_payments', 'exercised_stock_options', 'restricted_stock', 
              'total_stock_value', 'shared_receipt_with_poi']

clf= Pipeline(steps=[('scaler', scaler ), ('pca', pca), ('GNB', clf_GNB_opt)])

print test_classifier(clf, my_dataset, features_list, folds=1000)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
