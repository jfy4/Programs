#!/usr/bin/env python

import numpy as np
from itertools import product


def tgen(N):
    """
    Returns the infintesimal generators
    for SU(N) in the fundamental representation.

    Input:      N: the dimension of the group

    Output:     A numpy array of matricies in the
                fundamental representation.
    """

    def E(m, i, j):
        """
        Returns an m by m matrix with a 1 in
        the ith row, jth column.
        """
        temp = np.zeros((m, m))
        temp[i, j] = 1.0
        return temp

    def h(x):
        """
        Returns a x by x matrix with the x - 1 identity
        on the diagonal and minus the trace of that identity
        in the last diagonal entry, all normalized.
        """
        temp = np.zeros((x, x))
        temp[:(x - 1), :(x - 1)] = (np.sqrt(2.0 / (x * (x - 1))) *
                                    np.identity(x - 1))
        temp[x - 1, x - 1] = np.sqrt(2.0 / (x * (x - 1))) * (1 - x)
        return temp

    def hk(x, k):
        """
        Returns a x by x matrix with h(k) for the first k
        entries, and zeros otherwise.
        """
        temp = np.zeros((x, x))
        temp[:k, :k] = h(k)
        return temp

    t = []

    for j, k in product(range(N), range(N)):
        if (0 <= j < k < N):
            t.append(E(N, j, k) + E(N, k, j))
            t.append(-1j * E(N, j, k) + 1j * E(N, k, j))
    t.append(h(N))
    for i in range(2, N):
        t.append(hk(N, i))
    return np.array(t)
