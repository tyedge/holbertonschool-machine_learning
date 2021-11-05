#!/usr/bin/env python3
"""This module contains a function that calculates the probability density
function of a Gaussian distribution"""

import numpy as np


def pdf(X, m, S):
    """This function calculates the probability density function of a Gaussian
distribution"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(m) is not np.ndarray or len(m.shape) != 1:
        return None
    if type(S) is not np.ndarray or len(S.shape) != 2:
        return None

    d = X.shape[1]
    if S.shape[0] != d or S.shape[1] != d or m.shape[0] != d:
        return None
    det = np.linalg.det(S)
    inv = np.linalg.inv(S)
    num = np.exp(np.sum((np.matmul((X - m), np.linalg.inv(S)) * (X - m)),
                        axis=1) / -2)
    den = np.sqrt(det) * ((2 * np.pi) ** (d / 2))
    pdf = num / den
    pdf = np.where(pdf < 1e-300, 1e-300, pdf)
    return pdf
