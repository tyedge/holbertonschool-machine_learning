#!/usr/bin/env python3
"""This module contains a function that sets up Adam optimization for a keras
model with categorical crossentropy loss and accuracy metrics"""

import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """This function sets up Adam optimization for a keras model with
categorical crossentropy loss and accuracy metrics"""

    network.compile(optimizer=K.optimizers.Adam(alpha, beta1, beta2),
                    loss="categorical_crossentropy", metrics=["accuracy"])

    return None
