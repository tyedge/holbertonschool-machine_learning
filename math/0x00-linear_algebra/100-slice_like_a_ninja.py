#!/usr/bin/env python3
"""This module contains a function that slices a matrix along
a specific axes"""


def np_slice(matrix, axes={}):
    """This function slices a matrix along a specific axes"""
    matter = matrix.copy()

    ret = [slice(None)] * matter.ndim
    for key, val in axes.items():
        ret[key] = slice(*val)
    return matter[tuple(ret)]
