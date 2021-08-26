#!/usr/bin/env python3
"""This module contains a function that calculates the cost of a neural network
with L2 regularization"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """This function calculates the cost of a neural network with L2
regularization"""
    l2co = 0
    for i in range(1, L + 1):
        val = "W{}".format(str(i))
        l2co += np.linalg.norm(weights[val])
    return cost + ((lambtha / (2 * m)) * l2co)
