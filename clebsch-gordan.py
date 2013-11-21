from scipy.special import binom
from scipy.misc import factorial as fac
import numpy as np 

kroneker = lambda a, b: 1.0 if a == b else 0.0

def clebgor(j1, j2, j, m1, m2, m):
    zmin = int(min([j1 - m1, j2 + m2]))
    J = j1 + j2 + j
    return kroneker(m, m1 + m2) * np.sqrt(binom(2 * j1, J - 2 * j) * binom(2 * j2, J - 2 * j)/ \
            (binom(J + 1, J - 2 * j) * binom(2 * j1, j1 - m1) * \
            binom(2 * j2, j2 - m2) * binom(2 * j, j - m))) * \
            np.sum([(-1)**z * binom(J - 2 * j, z) * \
            binom(J - 2 * j2, j1 - m1 - z) * binom(J - 2 * j1, j2 + m2 - z) for z in range(zmin + 1)])
