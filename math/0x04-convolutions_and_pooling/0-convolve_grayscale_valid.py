#!/usr/bin/env python3
"""This module contains a function that performs a valid convolution on
grayscale images"""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """This funciton performs a valid convolution on grayscale images"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    oh = h - kh + 1
    ow = w - kw + 1
    out = np.zeros((m, oh, ow))
    for i in range(oh):
        for j in range(ow):
            out[:, i, j] = (kernel * images[:, i:i + kh, j:j + kw]).sum(
                axis=(1, 2))
    return out
