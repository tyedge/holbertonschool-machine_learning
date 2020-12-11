#!/usr/bin/env python3
"""This module contains a function that calculates the integral
of a polynomial"""


def poly_integral(poly, C=0):
    """This function calculates the integral of a polynomial"""
    if type(poly) is not list or poly == []:
        return None
    if poly == [0]:
        return [C]
    if type(C) is not int:
        return None

    res = []
    res.append(C)
    for x in range(len(poly)):
        inte = poly[x] / (x + 1)
        if inte.is_integer() is False:
            res.append(inte)
        else:
            res.append(int(inte))
    return res
