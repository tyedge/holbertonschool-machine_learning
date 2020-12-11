#!/usr/bin/env python3
"""This module contains a function that calculates the sum of the squares
of numbers i to n"""


def summation_i_squared(n):
    """This function returns the sum of the squares of numbers i to n"""
    if isinstance(n, int) is False or n < 1:
        return None
    sum = int((n * (n + 1) / 2) * (2 * n + 1) / 3)
    return sum
