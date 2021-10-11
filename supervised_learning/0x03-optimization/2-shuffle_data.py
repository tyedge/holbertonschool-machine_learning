#!/usr/bin/env python3
"""This module contains a function that shuffles the data points in two
matrices the same way"""

import numpy as np


def shuffle_data(X, Y):
    """This function shuffles the data points in two matrices the same way"""
    '''
    shuffler = np.random.permutation(X.shape[0])
    '''
    shuffler = np.random.permutation(X.shape[0])
    shufflex = X[shuffler]
    shuffley = Y[shuffler]
    return shufflex, shuffley
