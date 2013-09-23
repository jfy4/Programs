#!/usr/bin/env python

import sys
import os
import numpy as np 
import scipy as sp 
from itertools import product
from matplotlib import pyplot as plt 

def lincom(matrix):
    n=matrix.shape[0]
    one = np.identity(n**2)
    X = np.zeros((n, n, n**2))
    for i, j in product(range(n), range(n)):
        X[i,j,:] = one[:,n*i+j]
    C = (np.einsum('ia, ajk', matrix, X) - np.einsum('iak, aj', X, matrix)).reshape((n**2, n**2))
    B = np.zeros((n**2, n**2))
    XF = X.reshape((n**2, n**2))
    for i in range(n**2):
        B[i,:] = np.einsum('ia, a', C, XF[i,:])
    return B
