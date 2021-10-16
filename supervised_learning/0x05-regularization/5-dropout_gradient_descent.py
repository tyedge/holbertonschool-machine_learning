#!/usr/bin/env python3
"""This module contains a function that updates the weights of a neural network
with Dropout regularization using gradient descent"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """This function updates the weights of a neural network with Dropout
regularization using gradient descent"""

    sz = Y.shape[1]
    cpy = weights.copy()

    for i in range(L, 0, -1):
        a = "A{}".format(i)
        ap = cache["A{}".format(i - 1)]
        w = cpy.get("W{}".format(i))
        b = cpy.get("b{}".format(i))
        if i == L:
            diff = cache[a] - Y
        else:
            diff = np.matmul(cpy.get("W{}".format(i + 1)).T, dp) * (
                1 - (cache[a] ** 2))
        if i != L:
            diff = cache["D{}".format(i)] * diff / (keep_prob)

        dub = np.matmul(diff, ap.T) / sz
        bee = np.sum(diff, axis=1, keepdims=True) / sz
        weights["W{}".format(i)] -= (dub * alpha)
        weights["b{}".format(i)] -= (bee * alpha)
        dp = diff
