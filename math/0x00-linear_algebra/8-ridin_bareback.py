#!/usr/bin/env python3
"""This module contains a function that performs matrix multiplication"""


def mat_mul(mat1, mat2):
    """This function performs matrix multiplication"""
    m_mat1 = len(mat1)
    n_mat1 = len(mat1[0])
    m_mat2 = len(mat2)
    n_mat2 = len(mat2[0])

    if n_mat1 != m_mat2:
        return None

    matter = []
    while len(matter) < m_mat1:
        matter.append([])
        while len(matter[-1]) < n_mat2:
            matter[-1].append(0)

    for elem in range(m_mat1):
        for ent in range(n_mat2):
            sum = 0
            for a in range(n_mat1):
                sum += mat1[elem][a] * mat2[a][ent]
            matter[elem][ent] = sum
    return matter
