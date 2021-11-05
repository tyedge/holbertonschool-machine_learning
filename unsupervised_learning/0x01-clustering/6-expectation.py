#!/usr/bin/env python3
"""This module contains a function that calculates the expectation step in
the EM algorithm for a GMM"""

import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """This function calculates the expectation step in the EM algorithm
for a GMM"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if type(pi) is not np.ndarray or len(pi.shape) != 1:
        return None, None
    if type(m) is not np.ndarray or len(m.shape) != 2:
        return None, None
    if type(S) is not np.ndarray or len(S.shape) != 3:
        return None, None
    if not np.isclose(np.sum(pi), 1):
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    if (k, d) != m.shape or (k, d, d) != S.shape:
        return None, None

    fll = np.zeros((k, n))
    for i in range(k):
        prob = pdf(X, m[i], S[i])
        fll[i, :] = prob * pi[i]

    g = fll / np.sum(fll, axis=0)
    el = np.sum(np.log(np.sum(fll, axis=0)))

    return g, el
