#!/usr/bin/env python3
"""This module contains a function that tests for the optimum number of
clusters by variance"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """This function tests for the optimum number of clusters by variance"""

    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if kmax is None:
        kmax = X.shape[0]
    if type(kmin) is not int:
        return None, None
    if kmin < 1 or kmin >= X.shape[0]:
        return None, None
    if type(kmax) is not int:
        return None, None
    if kmax < 1 or kmax > X.shape[0]:
        return None, None
    if kmax <= kmin:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None

    results = []
    d_vars = []
    vlist = []
    for i in range(kmin, kmax + 1):
        C, cls = kmeans(X, i, iterations)
        results.append((C, cls))
        var = variance(X, C)
        vlist.append(var)
    for j in vlist:
        d_vars.append(vlist[0] - j)
    return results, d_vars
