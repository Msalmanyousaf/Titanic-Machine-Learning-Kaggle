# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 18:12:25 2020

@author: Salman
"""

import pandas as pd
import numpy as np
import pickle
from titanic_utils import *

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold

train = pd.read_csv('train.csv')
train_size = len(train)
test = pd.read_csv('test.csv')

# joining the train and test data for better feature analysis
data = pd.concat( [train, test], axis = 0 ).reset_index(drop = True)
data.fillna(np.nan, inplace = True)
original_features = list(train.columns)
original_features.remove('PassengerId')

train, train_y, test = data_preprocessing(data, train_size)


kfold = StratifiedKFold(n_splits=10)

######################### Random forest model ##########################
# RFC Parameters tunning 
# RFC = RandomForestClassifier()

# # Search grid for optimal parameters
# rf_param_grid = {"max_depth": [None],
#               "max_features": [1, 3, 10],
#               "min_samples_split": [2, 4, 10, 12, 16],
#               "min_samples_leaf": [1, 3, 10],
#               "bootstrap": [False],
#               "n_estimators" :[50, 100, 400, 700, 1000],
#               "criterion": ["gini", 'entropy']}


# gsRFC = GridSearchCV(estimator = RFC, param_grid = rf_param_grid, cv = kfold, 
#                       scoring = "accuracy", n_jobs = 4, verbose = 1)

# gsRFC.fit(train, train_y)

# RFC_best = gsRFC.best_estimator_

# # Best score
# rfc_best_score = gsRFC.best_score_


######################### Gradient Boosting model ##########################
# Gradient boosting tunning

GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1] 
              }

gsGBC = GridSearchCV(GBC, param_grid = gb_param_grid, cv = kfold, 
                     scoring = "accuracy", n_jobs = 4, verbose = 1)

gsGBC.fit(train, train_y)

GBC_best = gsGBC.best_estimator_

# Best score
gbc_best_score = gsGBC.best_score_


###################### Support Vector Machine model #######################

# SVC tuning
SVMC = SVC(probability=True)
svc_param_grid = {'kernel': ['rbf'], 
                  'gamma': [ 0.001, 0.01, 0.1, 1],
                  'C': [1, 10, 50, 100,200,300, 1000]}

gsSVMC = GridSearchCV(SVMC, param_grid = svc_param_grid, cv = kfold, 
                      scoring = "accuracy", n_jobs = 4, verbose = 1)

gsSVMC.fit(train, train_y)

SVMC_best = gsSVMC.best_estimator_

# Best score
svm_best_score = gsSVMC.best_score_

############################# Ensembling ###################################

voting = VotingClassifier (estimators = [('rfc', RFC_best), ('svc', SVMC_best), 
                                         ('gbc',GBC_best)], voting = 'soft', 
                                           n_jobs=4)

voting = voting.fit(train, train_y) 

pedictions = voting.predict(train)

######################### saving models ################################

pickle.dump(RFC_best, open('random_forest_best_model.sav', 'wb'))

pickle.dump(GBC_best, open('gradient_boosting_best_model.sav', 'wb'))

pickle.dump(SVMC_best, open('SVM_boosting_best_model.sav', 'wb'))

pickle.dump(voting, open('final_ensembled_model.sav', 'wb'))

