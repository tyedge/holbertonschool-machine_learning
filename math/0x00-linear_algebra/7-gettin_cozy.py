#!/usr/bin/env python3
"""This module contains a funciton that concatenates two matrices
along a specific axis"""


def cat_matrices2D(mat1, mat2, axis=0):
    """This function concatenates two matricies along a given axis"""
    m_mat1 = len(mat1)
    n_mat1 = len(mat1[0])
    m_mat2 = len(mat2)
    n_mat2 = len(mat2[0])
    matter = []
    lister = []
    arr = []

    if axis == 0 and n_mat1 != n_mat2:
        return None
    if axis == 1 and m_mat1 != m_mat2:
        return None

    elem = 0
    while elem in range(m_mat1):
        cp_mat1 = mat1[elem].copy()
        lister.append(cp_mat1)
        elem += 1

    ent = 0
    while ent in range(m_mat2):
        cp_mat2 = mat2[ent].copy()
        arr.append(cp_mat2)
        ent += 1

    if axis == 0 and len(lister[0]) == len(arr[0]):
        matter += (lister + arr)
        return matter

    if axis == 1 and len(lister) == len(arr):
        elem = 0
        while elem in range(len(lister)):
            matter.append(lister[elem] + arr[elem])
            elem += 1

    return matter
