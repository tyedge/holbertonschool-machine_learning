#!/usr/bin/env python3
"""This module contains a function that calculates a GMM from a dataset """

import sklearn.mixture


def gmm(X, k):
    """This function calculates a GMM from a dataset"""
    mix = sklearn.mixture.GaussianMixture(n_components=k).fit(X)
    pi = mix.weights_
    m = mix.means_
    S = mix.covariances_
    clss = mix.predict(X)
    bic = mix.bic(X)

    return pi, m, S, clss, bic
