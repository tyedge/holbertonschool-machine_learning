#!/usr/bin/env python3
"""This module contains a function that determines if you should stop gradient
descent early"""

import numpy as np


def early_stopping(cost, opt_cost, threshold, patience, count):
    """This function determines if you should stop gradient descent early"""
    if (threshold < (opt_cost - cost)):
        count = 0
        return False, count
    count += 1
    if (count == patience):
        return True, count
    return False, count
