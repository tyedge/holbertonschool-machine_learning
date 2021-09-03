#!/usr/bin/env python3
"""This module contains a function that performs a same convolution on
grayscale images"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """This function performs a same convolution on grayscale images"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph = int(kh / 2)
    pw = int(kw / 2)
    padder = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw)),
                    mode='constant', constant_values=0)
    out = np.zeros((images.shape))
    for i in range(h):
        for j in range(w):
            out[:, i, j] = (kernel * padder[:, i:i + kh, j:j + kw]).sum(
                axis=(1, 2))
    return out
