#!/usr/bin/env python3
"""This module contains a function that updates a variable in place using the
Adam optimization algorithm"""


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """This function updates a variable in place using the Adam optimization
algorithm"""

    gmom = beta1 * v + (1 - beta1) * grad
    grms = beta2 * s + (1 - beta2) * (grad ** 2)
    gmocorr = gmom / (1 - beta1 ** t)
    grmcorr = grms / (1 - beta2 ** t)
    varu = var - alpha * gmocorr / ((grmcorr ** 0.5) + epsilon)
    return varu, gmom, grms
