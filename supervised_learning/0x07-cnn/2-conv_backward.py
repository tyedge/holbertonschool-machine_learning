#!/usr/bin/env python3
"""This module contains a function that performs back propagation over a
convolutional layer of a neural network"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """This function performs back propagation over a convolutional layer of a
neural network"""
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == "valid":
        ph, pw = 0, 0

    if padding == "same":
        ph = int(((h_prev - 1) * sh - h_prev + kh) / 2) + 1
        pw = int(((w_prev - 1) * sw - w_prev + kw) / 2) + 1

    padder = np.pad(A_prev, pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    mode='constant', constant_values=0)

    dA_prev = np.zeros(padder.shape)
    dW = np.zeros(W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    oh = int((h_prev + 2 * ph - kh) / sh + 1)
    ow = int((w_prev + 2 * pw - kw) / sw + 1)

    out = np.zeros((m, oh, ow, c_new))

    for i in range(m):
        for j in range(h_new):
            for k in range(w_new):
                for n in range(c_new):
                    dA_prev[i, j * sh:j * sh + kh,
                            k * sw:k * sw + kw, :] += W[:, :, :, n] * dZ[
                                i, j, k, n]

                    dW[:, :, :, n] += padder[i, j * sh:j * sh + kh,
                                             k * sw:k * sw + kw, :] * dZ[
                                                 i, j, k, n]
    if padding is 'same':
        dA_prev = dA_prev[:, ph:dA_prev.shape[0]-ph, pw:dA_prev.shape[1]-pw, :]

    return dA_prev, dW, db
