#!/usr/bin/env python3
"""This module contains a function that builds an inception block as described
in Going Deeper with Convolutions (2014)"""

import tensorflow.keras as K


def inception_block(A_prev, filters):
    """This function builds an inception block as described in Going Deeper
with Convolutions (2014)"""
    init = K.initializers.he_normal()
    act = "relu"
    F1, F3R, F3, F5R, F5, FPP = filters
    dcon1 = K.layers.Conv2D(filters=F1, kernel_size=(1, 1), padding="same",
                            activation=act, kernel_initializer=init)(A_prev)
    dcon2 = K.layers.Conv2D(filters=F3R, kernel_size=(1, 1), padding="same",
                            activation=act, kernel_initializer=init)(A_prev)
    dcon3 = K.layers.Conv2D(filters=F3, kernel_size=(3, 3), padding="same",
                            activation=act, kernel_initializer=init)(dcon2)
    dcon4 = K.layers.Conv2D(filters=F5R, kernel_size=(1, 1), padding="same",
                            activation=act, kernel_initializer=init)(A_prev)
    dcon5 = K.layers.Conv2D(filters=F5, kernel_size=(5, 5), padding="same",
                            activation=act, kernel_initializer=init)(dcon4)
    pool = K.layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1),
                              padding="same")(A_prev)
    dcon6 = K.layers.Conv2D(filters=FPP, kernel_size=(1, 1), padding="same",
                            activation=act, kernel_initializer=init)(pool)
    out = K.layers.concatenate([dcon1, dcon3, dcon5, dcon6])
    return out
