#!/usr/bin/env python3
"""This module contains a function that performs forward propagation over a
pooling layer of a neural network"""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """This function performs forward propagation over a pooling layer of a
neural network"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    oh = int((h_prev - kh) / sh + 1)
    ow = int((w_prev - kw) / sw + 1)

    out = np.zeros((m, oh, ow, c_prev))

    for i in range(oh):
        for j in range(ow):
            if mode == "max":
                out[:, i, j, :] = np.max(A_prev[
                    :, i * sh:i * sh + kh, j * sw:j * sw + kw, :],
                                         axis=(1, 2))
            if mode == "avg":
                out[:, i, j, :] = np.mean(A_prev[
                    :, i * sh:i * sh + kh, j * sw:j * sw + kw, :],
                                          axis=(1, 2))
    return out
