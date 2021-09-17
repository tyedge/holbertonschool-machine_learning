#!/usr/bin/env python3
"""This module contains a function that builds the ResNet-50 architecture as
described in Deep Residual Learning for Image Recognition (2015)"""

import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """This function builds the ResNet-50 architecture as described in Deep
Residual Learning for Image Recognition (2015)"""
    data = K.Input(shape=(224, 224, 3))
    init = K.initializers.he_normal()

    conv = K.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2),
                           padding="same", kernel_initializer=init)(data)
    norm = K.layers.BatchNormalization(axis=3)(conv)
    act = K.layers.Activation("relu")(norm)
    pool = K.layers.MaxPool2D(pool_size=(3, 3), padding="same",
                              strides=(2, 2))(act)
    X = projection_block(pool, [64, 64, 256], 1)
    X = identity_block(X, [64, 64, 256])
    X = identity_block(X, [64, 64, 256])
    X = projection_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])
    X = projection_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = projection_block(X, [512, 512, 2048])
    X = identity_block(X, [512, 512, 2048])
    X = identity_block(X, [512, 512, 2048])
    avgpool = K.layers.AveragePooling2D(pool_size=(7, 7), padding="same",
                                        strides=None)(X)
    out = K.layers.Dense(units=1000, activation="softmax",
                         kernel_initializer=init)(avgpool)
    return K.Model(inputs=data, outputs=out)
