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
from sklearn.metrics import accuracy_score


def accuracy_score(expected, prediced):
    if type(expected) == type(prediced) == type(np.array([])):
        size = len(expected)
        score = 0.
        for i in range(size):
            if expected[i] == prediced[i]:
                score += 1
        return score/size


def find_best_classifier(data, target):
    x_train, x_test, y_train, y_test = \
            train_test_split(data, target,
                             test_size = 0.4,
                             random_state = 0)
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
    clf_svm.fit(x_train, y_train)

    clf_rf = GridSearchCV(
            RandomForestClassifier(),
            param[1].values()[0],
            n_jobs = -1)
    clf_rf.fit(x_train, y_train)

    return clf_svm, clf_rf


def clf_validation(data, target, clf, name):

    accuracy = 0.
    for j in range(10):
        x_train, x_test, y_train, y_test = \
                train_test_split(data, target,
                                 test_size = 0.4,
                                 random_state = j)

        y_predict = clf.predict(x_test)
        accuracy += accuracy_score(
                y_test,
                y_predict)/10

    print name + ' accuracy score :' + str(accuracy)


if __name__ == '__main__':
    a = np.array(['0', '1', '2', '3', '4', '5', '6'])
    b = np.array(['1', '1', '4', '3', '4', '6', '6'])
    print accuracy_score(a, b)
