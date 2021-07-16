#!/usr/bin/env python3
"""This program adds two matricies"""


def add_matrices2D(mat1, mat2):
    """This function returns the sum of two matricies"""
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None

    matter = []
    while len(matter) < len(mat1):
        matter.append([])
        while len(matter[-1]) < len(mat1[0]):
            matter[-1].append(0)

    for elem in range(len(mat1)):
        for ent in range(len(mat1[0])):
            matter[elem][ent] = mat1[elem][ent] + mat2[elem][ent]
    return matter
