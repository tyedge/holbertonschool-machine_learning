#!/usr/bin/env python3
"""This module contains a function that finds the best number of clusters
for a GMM using the Bayesian Information Criterion"""

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """This function finds the best number of clusters for a GMM using
the Bayesian Information Criterion"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None, None
    if kmax is None:
        kmax = X.shape[0]
    if kmin >= X.shape[0] or kmax > X.shape[0]:
        return None, None, None, None
    if type(kmin) is not int or kmin < 1:
        return None, None, None, None
    if type(kmax) is not int or kmax < 1:
        return None, None, None, None
    if kmin >= kmax:
        return None, None, None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None, None, None
    if type(tol) is not float or tol <= 0:
        return None, None, None, None
    if type(verbose) is not bool:
        return None, None, None, None

    n, d = X.shape
    bk, bres, lll, b = [], [], [], []

    for i in range(kmin, kmax + 1):
        bk.append(i)
        pi, m, S, t, v = expectation_maximization(X, i, iterations, tol,
                                                  verbose)
        bres.append((pi, m, S))
        lll.append(v)
        p = i - 1 + (d * i) + (i * d * (d + 1) / 2)
        bic = p * np.log(n) - 2 * v
        b.append(bic)

    best_k = bk[np.argmin(b)]
    best_result = bres[np.argmin(b)]
    lll = np.array(lll)
    b = np.array(b)

    return best_k, best_result, lll, b
