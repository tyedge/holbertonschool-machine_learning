#!/usr/bin/env python3
"""This module contains a function that converts a one-hot matrix into a vector
of labels"""

import numpy as np


def one_hot_decode(one_hot):
    """This function converts a one-hot matrix into a vector of labels"""
    if type(one_hot) is not np.ndarray or len(one_hot.shape) != 2:
        return None
    return np.argmax(one_hot, axis=0)
