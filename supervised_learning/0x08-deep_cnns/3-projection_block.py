#!/usr/bin/env python3
"""This module contains a function that builds a projection block as described
in Deep Residual Learning for Image Recognition (2015)"""

import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """This function builds a projection block as described in Deep Residual
Learning for Image Recognition (2015)"""
    F11, F3, F12 = filters
    init = K.initializers.he_normal()

    conv1 = K.layers.Conv2D(filters=F11, kernel_size=(1, 1), strides=(s, s),
                            padding="same", kernel_initializer=init)(A_prev)
    norm1 = K.layers.BatchNormalization(axis=3)(conv1)
    act1 = K.layers.Activation("relu")(norm1)
    conv2 = K.layers.Conv2D(filters=F3, kernel_size=(3, 3), padding="same",
                            kernel_initializer=init)(act1)
    norm2 = K.layers.BatchNormalization(axis=3)(conv2)
    act2 = K.layers.Activation("relu")(norm2)
    conv3 = K.layers.Conv2D(filters=F12, kernel_size=(1, 1), padding="same",
                            kernel_initializer=init)(act2)
    norm3 = K.layers.BatchNormalization(axis=3)(conv3)
    short = K.layers.Conv2D(filters=F12, kernel_size=(1, 1), strides=(s, s),
                            padding="same", kernel_initializer=init)(A_prev)
    sh_norm = K.layers.BatchNormalization(axis=3)(short)
    added = K.layers.Add()([norm3, sh_norm])
    out = K.layers.Activation("relu")(added)
    return out
