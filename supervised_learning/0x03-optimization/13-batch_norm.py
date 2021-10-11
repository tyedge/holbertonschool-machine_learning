#!/usr/bin/env python3
"""This module contains a funciton that normalizes an unactivated output of a
neural network using batch normalization"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """This function normalizes an unactivated output of a neural network
using batch normalization"""
    v = np.var(Z, axis=0)
    norm = (Z - np.mean(Z, axis=0)) / np.sqrt(v + epsilon)
    zret = norm * gamma + beta
    return zret
