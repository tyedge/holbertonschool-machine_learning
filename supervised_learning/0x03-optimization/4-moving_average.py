#!/usr/bin/env python3
"""This module contains a function that calculates the weighted moving average
of a data set"""

import numpy as np


def moving_average(data, beta):
    """This function calculates the weighted moving average of a data set"""
    avg = 0
    maverages = []
    for i in range(len(data)):
        avg = avg * beta + (1 - beta) * data[i]
        maverages.append(avg / (1 - (beta ** (i + 1))))
    return maverages
