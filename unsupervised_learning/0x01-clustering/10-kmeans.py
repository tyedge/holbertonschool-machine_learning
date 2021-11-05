#!/usr/bin/env python3
"""This module contains a function that performs K-means on a dataset"""

import sklearn.cluster


def kmeans(X, k):
    """This function performs K-means on a dataset"""
    km = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    C = km.cluster_centers_
    clss = km.labels_
    return C, clss
