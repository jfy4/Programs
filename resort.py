#!/usr/bin/env python

import numpy as np 
import itertools as it

def resort(matrix):
    """takes a (d**2, d) sized matrix and returns a (d, d, d) shaped tensor"""
    n = int(np.sqrt(matrix.shape[0]))
    out = np.zeros((n, n, matrix.shape[1]))
    for i,j,k in it.product(*[range(x) for x in out.shape]):
        out[i,j,k] = matrix[n*i+j, k]
    return out
