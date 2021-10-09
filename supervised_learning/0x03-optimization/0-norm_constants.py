#!/usr/bin/env python3
"""This module contains a function that calculates the normalization
(standardization) constants of a matrix"""

import numpy as np


def normalization_constants(X):
    """This function calculates the normalization (standardization) constants
of a matrix"""
    return np.mean(X, axis=0), np.std(X, axis=0)
