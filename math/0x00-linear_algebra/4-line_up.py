#!/usr/bin/env python3
"""This module contains a function that adds two arrays element-wise"""


def add_arrays(arr1, arr2):
    """This function adds two arrays element-wise"""
    if not len(arr1) == len(arr2):
        return None

    newer = []

    i = 0
    while i < len(arr1):
        newer.append(arr1[i] + arr2[i])
        i += 1
    return newer
