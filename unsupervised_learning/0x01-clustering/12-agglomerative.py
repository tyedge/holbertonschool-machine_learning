#!/usr/bin/env python3
"""This module contains a function that performs agglomerative clustering on a
dataset"""

import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """This function performs agglomerative clustering on a dataset"""
    Z = scipy.cluster.hierarchy.linkage(X, "ward")
    d = scipy.cluster.hierarchy.dendrogram(Z, color_threshold=dist)
    plt.show()
    return scipy.cluster.hierarchy.fcluster(Z, dist, criterion="distance")
