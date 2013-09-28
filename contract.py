#!/usr/bin/env python

import numpy as np 
import itertools as it

def contract(tensor):
    """Contracts the given tensor with itself"""
    temp = np.einsum('ikma, jlan', tensor, tensor)
    M = np.zeros((tensor.shape[0]**2, tensor.shape[1]**2, tensor.shape[2], tensor.shape[3]))
    for i,j,k,l,m,n in it.product(*[range(x) for x in temp.shape]):
        M[i + tensor.shape[0]*j, k + tensor.shape[1]*l, m, n] = temp[i,j,k,l,m,n]
    return M
