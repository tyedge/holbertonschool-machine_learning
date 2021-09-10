#!/usr/bin/env python3
"""This module contains a function that performs forward propagation over a
convolutional layer of a neural network"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """This function performs forward propagation over a convolutional layer of
a neural network"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == "valid":
        ph, pw = 0,0

    if padding == "same":
        ph = int(((h_prev - 1) * sh - h_prev + kh) / 2)
        pw = int(((w_prev - 1) * sw - w_prev + kw) / 2)

    padder = np.pad(A_prev, pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    mode='constant', constant_values=0)

    oh = int((h_prev + 2 * ph - kh) / sh + 1)
    ow = int((w_prev + 2 * pw - kw) / sw + 1)

    out = np.zeros((m, oh, ow, c_new))

    for i in range(oh):
        for j in range(ow):
            for n in range(c_new):
                out[:, i, j, n] = (W[:, :, :, n] * padder[
                    :, i * sh:i * sh + kh, j * sw:j * sw + kw, :]).sum(
                        axis=(1, 2, 3))
    return activation(out[:, i, j, n] + b[0, 0, 0, n])
