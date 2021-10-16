#!/usr/bin/env python3
"""This module contains a function that updates the weights and biases of a
neural network using gradient descent with L2 regularization"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """This function updates the weights and biases of a neural network using
gradient descent with L2 regularization"""

    sz = Y.shape[1]
    diff = cache["A{}".format(L)] - Y
    for i in range(L, 0, -1):
        nm = (lambtha / sz) * weights["W{}".format(i)]
        dub = ((1 / sz) * np.matmul(diff, cache["A{}".format(i - 1)].T)) + nm

        bee = (1 / sz) * np.sum(diff, axis=1, keepdims=True)

        weights["W{}".format(i)] -= (alpha * dub)
        weights["b{}".format(i)] -= (alpha * bee)

        d1 = np.matmul(weights["W{}".format(i)].T, diff)
        d2 = 1 - np.square(cache["A{}".format(i - 1)])
        diff = d1 * d2
