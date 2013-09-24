#!/usr/bin/env python

import sys
import os
import numpy as np 
import scipy as sp 
from itertools import product
from matplotlib import pyplot as plt 

def lincom(matrix):
    n = matrix.shape[0]
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

def hermx(matlist):
    tol = 1e-10
    n = np.shape(matlist[0])[0]
    matlistdag = np.array([np.transpose(np.conjugate(matlist[a])) for a in range(len(matlist))])
    T = np.array([lincom(matlist[a]) for a in range(matlist.shape[0])])
    Tdag = np.array([lincom(matlistdag[a]) for a in range(matlistdag.shape[0])])
    S = np.sum((np.einsum('ia, aj', np.transpose(np.conjugate(T[a])), T[a]) + np.einsum('ia, aj', np.transpose(np.conjugate(Tdag[a])), Tdag[a])) for a in range(len(T)))
    O = np.linalg.eig(S)
    num = np.sum(np.abs(O[0]) < tol)
    u = []
    for i in range(len(O[0])):
        if O[0][i] < tol:
            u.append(O[1][:,i])
        elif (np.abs(O[0]) > tol).all():
            raise ValueError("All eigenvalues greater than tolerance.")
    c = np.random.random((np.shape(u)[0]))
    c = c / np.sqrt(np.einsum('a, a', c, c))
    u = (np.sum(np.array(u)[a] * c[a] for a in range(len(c)))).reshape(n,n)
    return 0.5 * (u + np.transpose(np.conjugate(u)))