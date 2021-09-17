#!/usr/bin/env python3
"""This module contains a function that builds the inception network as
described in Going Deeper with Convolutions (2014)"""

import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """This function builds the inception network as described in Going Deeper
with Convolutions (2014)"""
    data = K.Input(shape=(224, 224, 3))
    init = K.initializers.he_normal()
    act = "relu"

    conv1 = K.layers.Conv2D(filters=64, kernel_size=(7, 7), padding="same",
                            activation=act, kernel_initializer=init,
                            strides=(2, 2))(data)
    pool1 = K.layers.MaxPool2D(pool_size=(3, 3), padding="same",
                               strides=(2, 2))(conv1)
    conv2 = K.layers.Conv2D(filters=64, kernel_size=(1, 1), padding="same",
                            activation=act, kernel_initializer=init,
                            strides=(1, 1))(pool1)
    conv3 = K.layers.Conv2D(filters=192, kernel_size=(3, 3), padding="same",
                            activation=act, kernel_initializer=init,
                            strides=(1, 1))(conv2)
    pool2 = K.layers.MaxPool2D(pool_size=(3, 3), padding="same",
                               strides=(2, 2))(conv3)
    incept1 = inception_block(pool2, [64, 96, 128, 16, 32, 32])
    incept2 = inception_block(incept1, [128, 128, 192, 32, 96, 64])
    padding = K.layers.MaxPool2D(pool_size=(3, 3), padding="same",
                                 strides=(2, 2))(incept2)
    incept3 = inception_block(padding, [192, 96, 208, 16, 48, 64])
    incept4 = inception_block(incept3, [160, 112, 224, 24, 64, 64])
    incept5 = inception_block(incept4, [128, 128, 256, 24, 64, 64])
    incept6 = inception_block(incept5, [112, 144, 288, 32, 64, 64])
    incept7 = inception_block(incept6, [256, 160, 320, 32, 128, 128])
    pool4 = K.layers.MaxPool2D(pool_size=(3, 3), padding="same",
                               strides=(2, 2))(incept7)
    incept8 = inception_block(pool4, [256, 160, 320, 32, 128, 128])
    incept9 = inception_block(incept8, [384, 192, 384, 48, 128, 128])
    avgpool = K.layers.AveragePooling2D(pool_size=(7, 7),
                                        strides=None, padding="same")(incept9)
    drop = K.layers.Dropout(rate=0.4)(avgpool)
    out = K.layers.Dense(units=1000, activation="softmax",
                         kernel_initializer=init)(drop)
    return K.Model(inputs=data, outputs=out)
