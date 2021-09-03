#!/usr/bin/env python3
"""This module contains a function that performs a convolution on grayscale
images with custom padding"""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """This function performs a convolution on grayscale images with custom
padding"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding
    padder = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw)),
                    mode='constant', constant_values=0)
    oh = h + (2 * ph) - kh + 1
    ow = h + (2 * pw) - kh + 1
    out = np.zeros((m, oh, ow))
    for i in range(oh):
        for j in range(ow):
            out[:, i, j] = (kernel * padder[:, i:i + kh, j:j + kw]).sum(
                axis=(1, 2))
    return out
