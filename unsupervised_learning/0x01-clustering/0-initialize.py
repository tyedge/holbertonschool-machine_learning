#!/usr/bin/env python3
"""This module contains a function that initializes cluster centroids for
K-means"""

import numpy as np


def initialize(X, k):
    """This function initializes cluster centroids for K-means"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(k) is not int or k <= 0:
        return None
    d = X.shape[1]
    min, max = X.min(axis=0), X.max(axis=0)
    return np.random.uniform(low=min, high=max, size=(k, d))
