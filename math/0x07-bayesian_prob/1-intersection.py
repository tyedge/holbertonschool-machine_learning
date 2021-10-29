#!/usr/bin/env python3
"""This module contains a funciton that calculates the likelihood and
intersection probability of obtaining this data with the various hypothetical
probabilities"""

import numpy as np


def likelihood(x, n, P):
    """This funciton that calculates the likelihood of obtaining this data
given various hypothetical probabilities of developing severe side effects"""

    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        raise ValueError("x must be an integer that is greater than or equal\
to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any(P > 1) or np.any(P < 0):
        raise ValueError("All values in P must be in the range [0, 1]")

    return ((np.math.factorial(n)) / (np.math.factorial(x) * (
        np.math.factorial(n - x))) * P ** x * (1 - P) ** (n - x))


def intersection(x, n, P, Pr):
    """This function calculates the intersection of obtaining this data with
the various hypothetical probabilities"""
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        raise ValueError("x must be an integer that is greater than or equal\
to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any(P > 1) or np.any(P < 0):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any(Pr > 1) or np.any(Pr < 0):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1.0):
        raise ValueError("Pr must sum to 1")

    return Pr * likelihood(x, n, P)
