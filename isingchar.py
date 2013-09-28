#!/usr/bin/env python

import numpy as np 

def IC(x):
    """Ising character"""
    if x == 0:
        return 1.0
    elif x == 1:
        return np.tanh(beta)
    else:
        return 0