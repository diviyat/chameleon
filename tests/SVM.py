#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 18:18:29 2020

@author: DiviyaT
"""
import numpy as np
import pandas as pd
from scipy import io
import timeit
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFE

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

def apply_SVM_RFE(X, y, **kwargs):
    n_features = kwargs.get('n_features', 1)
    step = kwargs.get('step', 1) 
    feature_subset = np.arange(X.shape[1])
    feature_idx_elimination_order = []
    for i in range(n_features, 0, -step):
        X_set = X[:, feature_subset]
        svc = LinearSVC(dual = False)
        rfe = RFE(svc, i, step=step, verbose=1)
        rfe.fit(X_set, y)
        boolean_mask = rfe.get_support(indices=False)
        pruned_feature_indices = feature_subset[np.invert(boolean_mask)]
        feature_idx_elimination_order.extend(list(pruned_feature_indices))
        feature_subset = feature_subset[boolean_mask]

    # Add the unpruned features 
    feature_idx_elimination_order.extend(feature_subset)
    return feature_idx_elimination_order[::-1], rfe
'''
start = timeit.default_timer()
print(apply_SVM_RFE(X=X, y=y,))
stop = timeit.default_timer()
timeTaken = stop - start

print('Time to run the loop: ', timeTaken)
'''