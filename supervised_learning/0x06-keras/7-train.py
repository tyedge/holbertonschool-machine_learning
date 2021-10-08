#!/usr/bin/env python3
"""This module contains a function that trains a model using mini-batch
gradient descent, analyzes validaiton data, trains the model using early
stopping, and trains the model with learning rate decay"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    """This function trains a model using mini-batch gradient descent,
analyzes validaiton data, trains the model using early stopping, and also
trains the model with learning rate decay"""

    if validation_data and early_stopping:
        callbcks = [K.callbacks.EarlyStopping(monitor='val_loss',
                                              patience=patience,
                                              mode="min")]
    if validation_data and learning_rate_decay:

        def scheduler(epoch):
            """This function takes an epoch index and returns a new
learning rate"""
            return alpha / (1 + decay_rate * epoch)

        callbcks = [K.callbacks.LearningRateScheduler(scheduler, verbose=1)]

    else:
        callbcks = None

    return network.fit(x=data, y=labels, batch_size=batch_size, epochs=epochs,
                       verbose=verbose, callbacks=callbcks,
                       validation_data=validation_data,
                       shuffle=shuffle)
