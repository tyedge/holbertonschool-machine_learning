#!/usr/bin/env python3
"""This module contains a function that tests a neural network"""

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """This function tests a neural network"""
    return network.evaluate(x=data, y=labels, verbose=verbose)
