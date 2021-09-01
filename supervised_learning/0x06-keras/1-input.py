#!/usr/bin/env python3
"""This module contains a function that builds a neural network with the Keras
library"""

import numpy as np
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """This function builds a neural network with the Keras library"""
    inlay = K.Input(shape=(nx,))
    outlay = K.layers.Dense(layers[0], activation=activations[0],
                            kernel_regularizer=K.regularizers.l2(
                                lambtha))(inlay)
    for i in range(len(layers)):
        if (i >= 1):
            dlay = K.layers.Dropout(1 - keep_prob)(outlay)
            outlay = K.layers.Dense(layers[i], activation=activations[i],
                                    kernel_regularizer=K.regularizers.l2(
                                        lambtha))(dlay)
    return K.Model(inlay, outlay)
