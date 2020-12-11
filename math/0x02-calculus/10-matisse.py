#!/usr/bin/env python3
"""This module contains a function which calculates the derivative
of a polynomial"""


def poly_derivative(poly):
    """This function returns the derivative of a polynomial"""
    if type(poly) is not list or len(poly) == 0:
        return None
    if len(poly) == 1:
        return [0]
    derv = []
    for i in range(len(poly)-1):
        if type(i) is not int:
            return None
        derv.append(poly[i + 1] * (i + 1))
    return derv
