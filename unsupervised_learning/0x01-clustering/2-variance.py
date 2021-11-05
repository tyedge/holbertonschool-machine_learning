#!/usr/bin/env python3
"""This module contains a function that calculates the total intra-cluster
variance for a data set"""

import numpy as np


def variance(X, C):
    """This function calculates the total intra-cluster variance for a
data set"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(C) is not np.ndarray or len(C.shape) != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None
    dist = np.linalg.norm(X - C[:, np.newaxis], axis=2)
    return np.sum(np.min(dist, axis=0)**2)
