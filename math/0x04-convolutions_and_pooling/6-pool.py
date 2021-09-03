#!/usr/bin/env python3
"""This module contains a function that performs pooling on images"""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """This function performs performs pooling on images"""
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    hsh = int(((h - kh) / sh) + 1)
    wsh = int(((w - kw) / sw) + 1)

    out = np.zeros((m, hsh, wsh, c))
    for i in range(hsh):
        for j in range(wsh):
            if mode == "max":
                out[np.arange(0, m), i, j] = np.max(images[np.arange(0, m),
                                                           i * sh:i * sh + kh,
                                                           j * sw:j * sw + kw],
                                                    axis=(1, 2))

            if mode == "avg":
                out[np.arange(0, m), i, j] = np.mean(images[np.arange(0, m),
                                                            i * sh:i * sh + kh,
                                                            j * sw:j * sw +
                                                            kw], axis=(1, 2))
    return out
