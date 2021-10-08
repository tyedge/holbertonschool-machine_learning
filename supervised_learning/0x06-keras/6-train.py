#!/usr/bin/env python3
"""This module contains a function that trains a model using mini-batch
gradient descent, analyzes validaiton data, and trains the model using
early stopping"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """This function trains a model using mini-batch gradient descent,
analyzes validaiton data, and also trains the model using early stopping"""

    if validation_data and early_stopping:
        callbcks = [K.callbacks.EarlyStopping(monitor='val_loss',
                                              patience=patience,
                                              mode="min")]
    else:
        callbcks = None
    return network.fit(x=data, y=labels, batch_size=batch_size, epochs=epochs,
                       verbose=verbose, callbacks=callbcks,
                       validation_data=validation_data,
                       shuffle=shuffle)
