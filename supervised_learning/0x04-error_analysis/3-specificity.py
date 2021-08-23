#!/usr/bin/env python3
"""This module contains a function that calculates the specificity for each
class in a confusion matrix"""

import numpy as np


def specificity(confusion):
    """This function calculates the specificity for each class in a confusion
matrix"""
    tp = np.array([confusion[i][i] for i in range(len(confusion))])
    fp = np.sum(confusion, axis=0) - tp
    tn = (np.sum(confusion) - np.sum(confusion, axis=1)) - fp
    return tn / (tn + fp)
