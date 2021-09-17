#!/usr/bin/env python3
"""This module contains a function that builds a dense block as described in
Densely Connected Convolutional Networks"""

import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """This function builds a dense block as described in Densely Connected
Convolutional Networks"""
    init = K.initializers.he_normal()
    for lay in range(layers):
        norm1 = K.layers.BatchNormalization()(X)
        act1 = K.layers.Activation("relu")(norm1)
        conv1 = K.layers.Conv2D(filters=(growth_rate * 4), kernel_size=(1, 1),
                                padding="same", kernel_initializer=init)(act1)

        norm2 = K.layers.BatchNormalization()(conv1)
        act2 = K.layers.Activation("relu")(norm2)
        conv2 = K.layers.Conv2D(filters=growth_rate, kernel_size=(3, 3),
                                padding="same", kernel_initializer=init)(act2)

        nb_filters += growth_rate
        X = K.layers.concatenate([conv2, X])

    return X, nb_filters
