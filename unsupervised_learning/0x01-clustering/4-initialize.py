#!/usr/bin/env python3
"""This module contains a function that initializes variables for a Gaussian
Mixture Model"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """This function initializes variables for a Gaussian Mixture Model"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None
    if type(k) is not int or k <= 0:
        return None, None, None
    if X.shape[0] < k:
        return None, None, None
    d = X.shape[1]
    m, n = kmeans(X, k)
    pi = np.ones((k)) / k
    S = np.full((k, d, d), np.eye(d))
    return pi, m, S
