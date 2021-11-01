#!/usr/bin/env python3
"""This module contains a function that performs PCA on a dataset"""

import numpy as np


def pca(X, var=0.95):
    """This function performs PCA on a dataset"""
    u, s, vh = np.linalg.svd(X)
    accum = np.cumsum(s) / np.sum(s)
    r = np.where(accum <= var, 1, 0)
    sumr = sum(r) + 1
    return vh.T[:, :sumr]
