#!/usr/bin/env python3
"""This module contains a function that performs back propagation over a
pooling layer of a neural network"""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """This function performs back propagation over a pooling layer of a neural
network"""
    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):
        for j in range(h_new):
            for k in range(w_new):
                for n in range(c_new):
                    if mode == "max":
                        p = A_prev[i, j * sh:j * sh + kh,
                                   k * sw:k * sw + kw, n]
                        dA_prev[i, j * sh:j * sh + kh,
                                k * sw:k * sw + kw, n] += (p == np.max(
                                    p)) * dA[i, j, k, n]

                    if mode == "avg":
                        dA_prev[i, j * sh:j * sh + kh,
                                k * sw:k * sw + kw, n] += dA[
                                    i, j, k, n] / (kh * kw)
    return dA_prev
