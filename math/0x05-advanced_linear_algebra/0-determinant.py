#!/usr/bin/env python3
"""This module contains a function that calculates the determinant of
a matrix"""


def determinant(matrix):
    """This function calculates the determinant of a matrix"""
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1

    for i in matrix:
        if type(i) is not list:
            raise TypeError("matrix must be a list of lists")

        if len(i) != len(matrix):
            raise ValueError("matrix must be a square matrix")

    if len(matrix) == 1:
        return matrix[0][0]

    if len(matrix) == 2:
        return (matrix[0][0] * matrix[1][1]) - (matrix[0][1]
                                                * matrix[1][0])
    deter = 0

    for j, k in enumerate(matrix[0]):
        rows = [r for r in matrix[1:]]
        sub = []
        for r in rows:
            sub.append([r[a] for a in range(len(matrix)) if a != j])
        deter += k * (-1) ** j * determinant(sub)
    return deter
