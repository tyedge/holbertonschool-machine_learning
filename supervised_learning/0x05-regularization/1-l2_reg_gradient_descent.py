#!/usr/bin/env python3
"""This module contains a function that updates the weights and biases of a
neural network using gradient descent with L2 regularization"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """This function updates the weights and biases of a neural network using
gradient descent with L2 regularization"""
    sz = Y.shape[1]
    cpy = weights.copy()
    dp = 0

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
        dub = np.matmul(diff, ap.T) / sz + ((lambtha / sz) * w)
        bee = np.sum(diff, axis=1, keepdims=True) / sz
        weights["W{}".format(i)] = w - (dub * alpha)
        weights["b{}".format(i)] = b - (bee * alpha)
        dp = diff
