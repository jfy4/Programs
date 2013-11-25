#!/usr/bin/env python

import numpy as np 
from itertools import product

def tgen(N):
    """
    Returns the infintesimal generators
    for SU(N) in the adjoint representation.

    Input:      N: the dimension of the group

    Output:     A numpy array of matricies in the
                adjoint representation. 
    """

    def E(m, i, j):
        """
        Returns an m by m matrix with a 1 in 
        the ith row, jth column.
        """
        temp = np.zeros((m, m))
        temp[i, j] = 1.0
        return temp

    t = []

    for j, k in product(range(N), range(N)):
        if (0 <= j < k < N):
            t.append(E(N, j, k) + E(N, k, j))
            t.append(-1j * E(N, j, k) + 1j * E(N, k, j))
        elif (j == k != (N - 1)):
            t.append(N * (E(N, k, k) - np.identity(N) / N))

    return np.array(t)