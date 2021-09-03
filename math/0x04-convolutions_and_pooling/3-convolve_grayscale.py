#!/usr/bin/env python3
"""This module contains a function that performs a convolution on grayscale
images"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """This function performs a convolution on grayscale images"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride
    if type(padding) is tuple:
        ph, pw = padding
    if padding is "valid":
        ph, pw = 0, 0
    if padding is "same":
        ph = int(((h - 1) * sh + kh - h) / 2) + 1
        pw = int(((w - 1) * sw + kw - w) / 2) + 1

    padder = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw)),
                    mode='constant', constant_values=0)

    oh = ((h - kh + 2 * ph) // sh) + 1
    ow = ((w - kw + 2 * pw) // sh) + 1

    out = np.zeros((m, oh, ow))
    for i in range(oh):
        for j in range(ow):
            out[:, i, j] = (kernel * padder[:, i * sh:i * sh + kh,
                                            j * sw:j * sw + kw]).sum(
                                                axis=(1, 2))
    return out
