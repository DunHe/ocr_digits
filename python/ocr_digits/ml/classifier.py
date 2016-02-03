# -*- coding: utf-8 -*-
'''
Create on 2016-01-18 10:29:14

@author: huangzhenghua
'''

import numpy as np
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV


def find_best_classifier(data, target):
    # x_train, x_test, y_train, y_test = \
            # train_test_split(data, target,
                             # test_size = 0.4,
                             # random_state = 0)
    param = [
        {'svm': [
            {
                'kernel': ['rbf'],
                'C': [1, 10, 100, 1000],
                'gamma': [1e-3, 1e-4]
            },
            {
                'kernel': ['linear', 'sigmoid'],
                'C': [1, 10, 100, 1000]
            }
        ]}, 
        {'rf': [
            {
                'n_estimators': [10, 100, 300],
                'max_features': ['auto', 'log2']
            }
        ]},
    ]

    clf_svm = GridSearchCV(
            SVC(),
            param[0].values()[0],
            n_jobs = -1)
    # clf_svm.fit(x_train, y_train)
    clf_svm.fit(data, target)

    clf_rf = GridSearchCV(
            RandomForestClassifier(),
            param[1].values()[0],
            n_jobs = -1)
    # clf_rf.fit(x_train, y_train)
    clf_rf.fit(data, target)

    return clf_svm, clf_rf
