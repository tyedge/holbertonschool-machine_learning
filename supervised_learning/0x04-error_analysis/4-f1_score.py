#!/usr/bin/env python3
"""This module contains a function that calculates the F1 score of a confusion
matrix"""

import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """This function calculates the F1 score of a confusion matrix"""
    top = precision(confusion) * sensitivity(confusion)
    bot = precision(confusion) + sensitivity(confusion)
    return 2 * top / bot
