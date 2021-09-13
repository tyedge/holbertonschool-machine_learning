#!/usr/bin/env python3
"""This module contains a funtion that builds a modified version of the LeNet-5
architecture using keras"""

import tensorflow.keras as K


def lenet5(X):
    """This function builds a modified version of the LeNet-5 architecture
using keras"""
    init = K.initializers.he_normal()
    relu = "relu"

    conv1 = K.layers.Conv2D(filters=6,
                            kernel_size=(5, 5),
                            padding="same",
                            activation=relu,
                            kernel_initializer=init)(X)

    pool1 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)

    conv2 = K.layers.Conv2D(filters=16,
                            kernel_size=(5, 5),
                            padding="valid",
                            activation=relu,
                            kernel_initializer=init)(pool1)

    pool2 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)

    flat = K.layers.Flatten()(pool2)

    full1 = K.layers.Dense(units=120,
                           activation=relu,
                           kernel_initializer=init)(flat)

    full2 = K.layers.Dense(units=84,
                           activation=relu,
                           kernel_initializer=init)(full1)

    fullout = K.layers.Dense(units=10, activation="softmax",
                             kernel_initializer=init)(full2)

    kmod = K.Model(X, fullout)
    kmod.compile(optimizer=K.optimizers.Adam(),
                 loss="categorical_crossentropy",
                 metrics=["accuracy"])

    return kmod
