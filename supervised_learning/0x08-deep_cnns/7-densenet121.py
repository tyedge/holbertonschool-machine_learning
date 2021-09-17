#!/usr/bin/env python3
"""This module contains a function that builds the DenseNet-121 architecture
as described in Densely Connected Convolutional Networks"""

import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """This function builds the DenseNet-121 architecture as described in
Densely Connected Convolutional Networks"""
    init = K.initializers.he_normal()
    data = K.Input(shape=(224, 224, 3))

    norm = K.layers.BatchNormalization()(data)
    act = K.layers.Activation("relu")(norm)
    conv = K.layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2),
                           padding="same", kernel_initializer=init)(act)
    pool = K.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2),
                              padding="same")(conv)
    X, nb_filters = dense_block(pool, 64, growth_rate, 6)
    X, nb_filters = transition_layer(X, nb_filters, compression)
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 12)
    X, nb_filters = transition_layer(X, nb_filters, compression)
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 24)
    X, nb_filters = transition_layer(X, nb_filters, compression)
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 16)

    avgpool = K.layers.AveragePooling2D(pool_size=(7, 7), strides=(1, 1))(X)
    out = K.layers.Dense(1000, activation="softmax")(avgpool)

    return K.Model(inputs=data, outputs=out)
