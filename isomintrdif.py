#!/usr/bin/env python

import numpy as np 
import itertools as it

def isomintrdif(tensor):
    """Updates a tensor attempting to minimize the difference between
    the trucated tensor trace and the original."""
    d = tensor.shape
    temp = np.einsum('ikma, jlan', tensor, tensor)
    M = np.zeros((d[0]**2, d[1]**2, d[2], d[3]))
    for i,j,k,l,m,n in it.product(*[range(x) for x in temp.shape]):
        M[d[0]*i+j, d[1]*k+l, m, n] = temp[i,j,k,l,m,n]
    q = np.einsum('ijaa', M)
    O = np.linalg.eig(np.einsum('ia,aj', q, q))
    tot = np.sum(O[0])
    ee = []
    for i in it.combinations(zip(O[0], np.transpose(O[1])), ns):
        moe = zip(*i)
        ee.append((np.abs(np.sum(moe[0]) - tot), np.transpose(moe[1])))
    moe = zip(*ee)
    U = moe[1][np.argmin(moe[0])]
    temp = np.einsum('ajkl, ai', M, U)
    U = resort(U)
    temp = np.einsum('iakm, jaln', temp, temp)
    return np.einsum('ijkab, abl', np.einsum('ijablm, abk', temp, U), U)
