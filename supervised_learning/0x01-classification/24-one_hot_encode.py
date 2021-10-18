#!/usr/bin/env python3
"""This module contains a function that converts a numeric label vector into a
one-hot matrix"""

import numpy as np


def one_hot_encode(Y, classes):
    """This function converts a numeric label vector into a one-hot matrix"""
    if type(Y) is not np.ndarray or len(Y) < 1:
        return None
    if type(classes) is not int or classes <= np.amax(Y):
        return None
    one_hot = np.zeros((classes, Y.shape[0]))
    c = np.arange(Y.size)
    one_hot[Y, c] = 1
    return one_hot
