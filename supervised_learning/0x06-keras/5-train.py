#!/usr/bin/env python3
"""This module contains a function that trains a model using mini-batch
gradient descent and also analyzes validaiton data"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """This function trains a model using mini-batch gradient descent
and also analyzes validaiton data"""

    return network.fit(x=data, y=labels, batch_size=batch_size, epochs=epochs,
                       verbose=verbose, validation_data=validation_data,
                       shuffle=shuffle)
