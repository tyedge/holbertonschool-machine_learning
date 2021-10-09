#!/usr/bin/env python3
"""This module contains functions that finds the shape of and adds two
matrices"""


def matrix_shape(matrix):
    """This function calculates the shape of a matrix"""

    if type(matrix) != list:
        return []
    return [len(matrix)] + matrix_shape(matrix[0])


def extender(mat1, mat2, ax1, ax2):
    """Test 1"""

    ret = []
    if ax1 != ax2:
        for i in range(len(mat1)):
            ret.append(extender(mat1[i], mat2[i], ax1, ax2 + 1))
        return ret
    else:
        ret = mat1 + mat2
        return ret


def cat_matrices(mat1, mat2, axis=0):
    """This function concatenates two matrices along a specific axis"""
    a = matrix_shape(mat1)
    b = matrix_shape(mat2)
    del a[axis]
    del b[axis]
    if a != b:
        return None
    if axis >= len(a) + 1:
        return None
    ret = extender(mat1, mat2, axis, 0)
    return ret
