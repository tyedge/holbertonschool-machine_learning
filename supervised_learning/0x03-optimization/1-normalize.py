#!/usr/bin/env python3
"""This module contains a function that normalizes (standardizes) a matrix"""


def normalize(X, m, s):
    """This function normalizes (standardizes) a matrix"""
    return (X - m) / s
