#!/usr/bin/env python3
"""This module contains a function that performs K-means on a dataset"""

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


def kmeans(X, k, iterations=1000):
    """This function performs K-means on a dataset"""
    if type(iterations) is not int or iterations < 1:
        return None, None

    cent = initialize(X, k)
    if cent is None:
        return None, None

    for i in range(iterations):
        cpy = np.copy(cent)
        dist = np.linalg.norm(X - cent[:, np.newaxis], axis=2)
        cls = np.argmin(dist, axis=0)

        for j in range(k):
            if len(X[j == cls]) == 0:
                cent[j] = initialize(X, 1)
            else:
                cent[j] = np.mean(X[j == cls], axis=0)
        if (cpy == cent).all():
            return cent, cls

    return cent, cls
