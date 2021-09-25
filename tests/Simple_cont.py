#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 18:16:19 2020

@author: DiviyaT
"""

import numpy as np
import os
import jpype as jp
from scipy.stats import t
import pathlib
import pandas as pd
from scipy import io
import timeit

in_file = io.loadmat('ALLAML.mat')
X = pd.DataFrame(in_file['X'], dtype=float)
y = pd.DataFrame(in_file['Y'])


# convert classes from whatever datatype they are to binary integers (0 and 1)
y_values = np.unique(y)
if len(y_values) > 2:
    raise errors.NonBinaryTargets()
y_binary = np.array(y == y_values[0], dtype=int)

X = X.values

y = np.reshape(y_binary, -1)

#y = y.reshape(-1) ##RESHAPE

legacy_dir = pathlib.Path(__file__).parent.absolute()
jarLocation = legacy_dir / "infodynamics.jar" 

"""Global property defaults"""
# Calculator properties
mi_class_types = {
    'kraskov': 'infodynamics.measures.mixed.kraskov.MutualInfoCalculatorMultiVariateWithDiscreteKraskov',
    'kraskov_cont': 'infodynamics.measures.continuous.kraskov.MutualInfoCalculatorMultiVariateKraskov1'
}
cond_mi_class_types = {
    'kraskov': 'infodynamics.measures.mixed.kraskov.ConditionalMutualInfoCalculatorMultiVariateWithDiscreteKraskov',
    'kraskov_cont': 'infodynamics.measures.continuous.kraskov.ConditionalMutualInfoCalculatorMultiVariateKraskov1'
}
mi_calc_properties = {
    'normalise': 'false',
    'k': '4',
    'norm_type': 'max_norm'
}

# Kernel properties
num_samples = 1000
p_value_threshold = 0.05
max_joint_vars_in_iterative = 5
kNNsk = [1, 3, 5, 7, 9, 11, 13]

def binarise_y(y):
    y_vals = np.unique(y)
    for i in range(y.shape[0]):
        y[i] = 0 if y[i] == y_vals[0] else 1
    return y


def simple_MI(X, y, **kwargs):
    if not jp.isJVMStarted():
        jp.startJVM(jp.getDefaultJVMPath(), "-ea", "-Djava.class.path=" + str(jarLocation))

    # Set properties from kwargs
    mi_calc_type = kwargs.get('mi_calc_type', 'kraskov_cont')
    mi_class = mi_class_types[mi_calc_type]
    cond_mi_class = cond_mi_class_types[mi_calc_type]
    # Initialise calculator
    mi_calc_class = jp.JClass(mi_class)
    cond_mi_calc_class = jp.JClass(cond_mi_class)
    mi_calc = mi_calc_class()
    cond_mi_calc = cond_mi_calc_class()
    for prop, value in mi_calc_properties.items():
        mi_calc.setProperty(jp.JString(prop), jp.JString(value))
        cond_mi_calc.setProperty(prop, value)
    y = y.reshape(-1) #RESHAPE
    y = binarise_y(y)

    # Set up calculation variables 
    MIs = []
    
    # Perform feature selection... feature added in each iteration
    for cand_idx in range(0,X.shape[1]):
        mi_calc.initialise(1,1) #UNIVARIATE INITIALISE 1,1
        X1 = X[:,cand_idx]
        X2 = X1.tolist() ## TO LIST TO CONVERT FLOAT64 TO FLOAT --> then convert to java double
        y1 = y.tolist()
        X_j = jp.JArray(jp.JDouble, 1)(X2)
        y_j = jp.JArray(jp.JDouble, 1)(y1)
        mi_calc.setObservations(X_j, y_j)
        mi = mi_calc.computeAverageLocalOfObservations()
        
        MIs.append(mi)
        
    sorted_indices = np.argsort(MIs)[::-1]
    sorted_MIs = np.sort(MIs)[::-1]

    return sorted_indices

'''
start = timeit.default_timer()
print(simple_MI(X=X, y=y,))
stop = timeit.default_timer()
timeTaken = stop - start

print('Time to run the loop: ', timeTaken)
'''








