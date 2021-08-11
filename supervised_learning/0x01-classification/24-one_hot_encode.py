#!/usr/bin/env python3
"""This module contains a function that converts a numeric label vector into a
one-hot matrix"""

import numpy as np


def one_hot_encode(Y, classes):
    """This function converts a numeric label vector into a one-hot matrix"""
    if type(Y) is not np.ndarray:
        return None
    elif len(Y) <= 0:
        return None
    elif type(classes) is not int:
        return None
    elif classes <= np.amax(Y):
        return None
    else:
        return np.eye(classes)[Y].T
