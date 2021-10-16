#!/usr/bin/env python3
"""This module contains a function that updates the weights and biases of a
neural network using gradient descent with L2 regularization"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """This function updates the weights and biases of a neural network using
gradient descent with L2 regularization"""
    sz = Y.shape[1]
    cpy = weights.copy()
    for i in reversed(range(L)):
        w = "W{}".format(i + 1)
        b = "b{}".format(i + 1)
        a1 = "A{}".format(i + 1)
        a2 = "A{}".format(i)
        nrm = (lambtha / sz) * cpy[w]
        if i == (L - 1):
            diff = cache[a1] - Y
            dub = (np.matmul(cache[a2], diff.T) / sz).T
        else:
            d1 = np.matmul(cpy["W{}".format(i + 2)].T, diff)
            d2 = 1 - cache[a1] ** 2
            diff = d1 * d2
        dub = np.matmul(diff, cache[a2].T) / sz + nrm
        bee = np.sum(diff, axis=1, keepdims=True) / sz
        weights[w] -= (alpha * dub)
        weights[b] -= (alpha * bee)
