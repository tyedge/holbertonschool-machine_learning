#!/usr/bin/env python3
"""This module contains a funciton that makes a prediction using a
neural network"""

import tensorflow.keras as K


def predict(network, data, verbose=False):
    """This function makes a prediction using a neural network"""
    return network.predict(x=data, verbose=verbose)
