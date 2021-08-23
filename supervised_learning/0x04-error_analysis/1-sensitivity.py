#!/usr/bin/env python3
"""This module contains a function that calculates the sensitivity for each
class in a confusion matrix"""

import numpy as np


def sensitivity(confusion):
    """This function calculates the sensitivity for each class in a confusion
matrix"""
    tp = np.array([confusion[i][i] for i in range(len(confusion))])
    return tp / np.sum(confusion, axis=1)
