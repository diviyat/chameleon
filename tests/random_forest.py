#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 23:16:46 2020

@author: DiviyaT
"""

import numpy as np
import pandas as pd
from scipy import io
import timeit
from sklearn.ensemble import RandomForestClassifier

in_file = io.loadmat('colon.mat')
X = pd.DataFrame(in_file['X'], dtype=float)
y = pd.DataFrame(in_file['Y'])

# convert classes from whatever datatype they are to binary integers (0 and 1)
y_values = np.unique(y)
if len(y_values) > 2:
    raise errors.NonBinaryTargets()
y_binary = np.array(y == y_values[0], dtype=int)

X = X.values

y = np.reshape(y_binary, -1)

def apply_RF(X, y, **kwargs):
    n_estimators = kwargs.get('n_estimators', 100)
    rf = RandomForestClassifier(n_estimators=n_estimators)
    rf.fit(X, y)
    selected_features = np.argsort(rf.feature_importances_)[::-1]
    return selected_features

'''
start = timeit.default_timer()
print(apply_RF(X=X, y=y,))
stop = timeit.default_timer()
timeTaken = stop - start

print('Time to run the loop: ', timeTaken)
'''