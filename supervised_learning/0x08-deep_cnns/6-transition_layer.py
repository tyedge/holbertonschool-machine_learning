#!/usr/bin/env python3
"""This module contains a function that builds a transition layer as described
in Densely Connected Convolutional Networks"""

import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """This function builds a transition layer as described in Densely
Connected Convolutional Networks"""
    init = K.initializers.he_normal()
    fltrs = int(compression * nb_filters)

    norm = K.layers.BatchNormalization()(X)
    act = K.layers.Activation("relu")(norm)
    conv = K.layers.Conv2D(filters=fltrs, kernel_size=(1, 1), padding="same",
                           kernel_initializer=init)(act)
    avgpool = K.layers.AveragePooling2D(pool_size=(2, 2), padding="same")(conv)

    return avgpool, fltrs
