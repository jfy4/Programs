#!/usr/bin/env python

import numpy as np 
from itertools import product

def isodynnumstates(tensor):
    """Takes a tensor.  Contract the tensor with itself and 
    reduces the d**2 product-state legs to size d. This 
    changes the states in order to meet a tolerance"""
    d = tensor.shape
    temp = np.einsum('ikma, jlan', tensor, tensor)
    M = np.zeros((d[0]**2, d[1]**2, d[2], d[3]))
    for i,j,k,l,m,n in product(*[range(x) for x in temp.shape]):
        M[d[0]*i+j, d[1]*k+l, m, n] = temp[i,j,k,l,m,n]
    U, s, _ = np.linalg.svd(np.einsum('ijaa', M)) # note the contraction pattern
    tot = np.sum(s)
    for i in range(len(s)):
        if (np.sum(s[a] for a in range(i+1))/tot >= tol):
            ns = i + 1
            break
    temp = np.einsum('ajkl, ai', M, U[:, :ns])
    temp = np.einsum('iakm, jaln', temp, temp)
    return np.einsum('ijkab, abl', np.einsum('ijablm, abk', temp, resort(U[:, :ns])), resort(U[:, :ns]))
