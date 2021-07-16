#!/usr/bin/env python3
"""This module contains a function that performs elementwise
arithmetic operations on matricies"""


def np_elementwise(mat1, mat2):
    """This function performs arithmetic operations on the
    elements of a matrix"""
    add = mat1 + mat2
    sub = mat1 - mat2
    mul = mat1 * mat2
    div = mat1 / mat2

    return(add, sub, mul, div)
