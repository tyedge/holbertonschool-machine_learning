#!/usr/bin/env python3
"""This module contains a function that tests for the optimum number of
clusters by variance"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """This function tests for the optimum number of clusters by variance"""

    if type(X) is not np.ndarray:
        return None, None
    if type(kmin) is not int:
        return None, None
    if kmax and type(kmax) is not int:
        return None, None
    if kmax and kmax <= kmin:
        return None, None
    if not kmax:
        kmax = X.shape[0]
    if type(iterations) is not int:
        return None, None
    if iterations <= 0:
        return None, None
    if len(X.shape) != 2:
        return None, None
    if kmin < 1:
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
        d_vars.append(vlist[0] - float(j))
    return results, d_vars
