#!/usr/bin/env python3
"""This module contains a function that performs PCA on a dataset"""

import numpy as np


def pca(X, ndim):
    """This function performs PCA on a dataset"""
    u, s, vh = np.linalg.svd(X - np.mean(X, axis=0))
    T = np.matmul(X - np.mean(X, axis=0), vh[:ndim].T)
    return T
