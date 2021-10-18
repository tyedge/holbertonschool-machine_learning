#!/usr/bin/env python3
"""This module contains a function that conducts forward propagation using
Dropout"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """This function conducts forward propagation using Dropout"""
    cache = {}
    cache['A0'] = X

    for i in range(L):
        weight = "W{}".format(str(i + 1))
        bias = "b{}".format(str(i + 1))
        ckey0 = "A{}".format(str(i))
        ckey1 = "A{}".format(str(i + 1))
        ckey2 = "D{}".format(str(i + 1))
        z = np.matmul(weights[weight], cache[ckey0]) + weights[bias]
        d = np.random.binomial(1, p=keep_prob, size=z.shape)
        if i == L - 1:
            cache[ckey1] = (np.exp(z)/np.sum(np.exp(z), axis=0, keepdims=True))
        else:
            cache[ckey1] = np.tanh(z)
            cache[ckey2] = d
            cache[ckey1] *= d
            cache[ckey1] = cache[ckey1] / keep_prob
    return cache
