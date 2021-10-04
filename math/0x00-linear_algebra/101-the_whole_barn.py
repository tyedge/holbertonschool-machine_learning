#!/usr/bin/env python3
"""This module contains functions that finds the shape of and adds two
matrices"""


def matrix_shape(matrix):
    """This function calculates the shape of a matrix"""
    if type(matrix) != list:
        return []
    return [len(matrix)] + matrix_shape(matrix[0])


def add_matrices(mat1, mat2):
    """This function adds two matrices"""
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None
    ret = []
    if type(mat1[0]) is list:
        for x, y in zip(mat1, mat2):
            ret.append(add_matrices(x, y))
        return ret
    else:
        [ret.append((mat1[i] + mat2[i])) for i in range(len(mat1))]
        return ret
