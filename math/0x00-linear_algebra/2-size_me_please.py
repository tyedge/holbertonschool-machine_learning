#!/usr/bin/env python3
"""This module contains a function that calculates the shape
of a matrix"""


def matrix_shape(matrix):
    """This function calculates the shape of a matrix"""
    if type(matrix) != list:
        return []
    return [len(matrix)] + matrix_shape(matrix[0])
