#!/usr/bin/env python3
"""This module contains a function that creates a confusion matrix"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """This function creates a confusion matrix"""
    out = np.matmul(labels.T, logits)
    return out
