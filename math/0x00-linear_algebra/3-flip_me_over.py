#!/usr/bin/env python3
"""This module contains a function that transposes a 2D matrix"""


def matrix_transpose(matrix):
    """This function transposes a 2D matrix"""
    m = len(matrix)
    n = len(matrix[0])
    matter = []
    while len(matter) < n:
        matter.append([])
        while len(matter[-1]) < m:
            matter[-1].append(0)
    for elem in range(m):
        for ent in range(n):
            matter[ent][elem] = matrix[elem][ent]
    return matter
