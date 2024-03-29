#!/usr/bin/env python3
"""This module contains a function that performs the expectation maximization
for a GMM"""

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """This function performs the expectation maximization for a GMM"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None, None, None
    if type(k) is not int or k < 1:
        return None, None, None, None, None
    if type(iterations) is not int or iterations < 1:
        return None, None, None, None, None
    if type(tol) is not float or tol < 0:
        return None, None, None, None, None
    if type(verbose) is not bool:
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    g, lky = expectation(X, pi, m, S)
    act = 0

    for i in range(iterations):
        if verbose and i % 10 == 0:
            print("Log Likelihood after {} iterations: {}".format(
                i, lky.round(5)))
        pi, m, S = maximization(X, g)
        g, lky = expectation(X, pi, m, S)

        if abs(act - lky) <= tol:
            break
        act = lky

    i += 1
    if verbose:
        print("Log Likelihood after {} iterations: {}".format(i, lky.round(5)))

    return pi, m, S, g, lky
