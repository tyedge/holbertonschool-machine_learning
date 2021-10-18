#!/usr/bin/env python3
"""This module contains a function that updates the weights of a neural network
with Dropout regularization using gradient descent"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """This function updates the weights of a neural network with Dropout
regularization using gradient descent"""

    cpy = weights.copy()
    sz = Y.shape[1]
    for i in reversed(range(L)):
        if i == L - 1:
            diff = cache["A{}".format(i + 1)] - Y
            dub = (np.matmul(cache["A{}".format(i)], diff.T) / sz).T
        else:
            d1 = np.matmul(cpy["W{}".format(i + 2)].T, dp)
            d2 = 1 - cache["A{}".format(i+1)] ** 2
            diff = d1 * d2
            diff *= cache["D{}".format(i + 1)]
            diff /= keep_prob
            dub = np.matmul(diff, cache["A{}".format(i)].T) / sz
        bee = np.sum(diff, axis=1, keepdims=True) / sz
        weights["W{}".format(i + 1)] = cpy["W{}".format(i + 1)] - alpha * dub
        weights["b{}".format(i + 1)] = cpy["b{}".format(i + 1)] - alpha * bee
        dp = diff
