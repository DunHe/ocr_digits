# -*- coding: utf-8 -*-
'''
Create on 2016-01-27 19:00:01

@author: huangzhenghua
'''

import numpy as np
from sklearn.cross_validation import train_test_split


def accuracy_score(expected, prediced):

    if type(expected) == type(prediced) == type(np.array([])):
        size = len(expected)
        score = 0.
        for i in range(size):
            if expected[i] == prediced[i]:
                score += 1
        return score/size


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
