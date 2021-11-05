#!/usr/bin/env python3
"""This module contains a function that calculates the maximization step
in the EM algorithm for a GMM"""

import numpy as np


def maximization(X, g):
    """This funciton calculates the maximization step in the EM algorithm
for a GMM"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None
    if type(g) is not np.ndarray or len(g.shape) != 2:
        return None, None, None
    n, d = X.shape
    gs = g.shape[0]
    if n != g.shape[1]:
        return None, None, None
    if not np.isclose(np.sum(g, axis=0), np.ones((n,))).all():
        return None, None, None
    pi = (1 / n) * g.sum(axis=1)
    m = np.matmul(g, X)
    S = np.zeros((gs, d, d))
    for i in range(gs):
        m[i] /= g.sum(axis=1)[i]
        S[i] = np.matmul(g[i].reshape(1, n) * (X - m[i]).T, (X - m[i]))
        S[i] /= g.sum(axis=1)[i]
    return pi, m, S
