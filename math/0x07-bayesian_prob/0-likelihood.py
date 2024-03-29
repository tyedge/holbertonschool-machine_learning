#!/usr/bin/env python3
"""This module contains a funciton that calculates the likelihood of obtaining
this data given various hypothetical probabilities of developing severe side
effects"""

import numpy as np


def likelihood(x, n, P):
    """This funciton that calculates the likelihood of obtaining this data
given various hypothetical probabilities of developing severe side effects"""
    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) != int or x < 0:
        raise ValueError("x must be an integer that is greater than or equal\
to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if np.any(P > 1) or np.any(P < 0):
        raise ValueError("All values in P must be in the range [0, 1]")

    ret = (np.math.factorial(n) / (np.math.factorial(x) * np.math.factorial(
        n - x))) * (P ** x) * ((1 - P) ** (n - x))

    return ret
