#!/usr/bin/env python3
"""This module contains two functions: one that saves a model’s weights and
another which loads a model’s weights"""


import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """This function saves a model’s weights"""
    network.save_weights(filepath=filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """This function loads a model's weights"""
    network.load_weights(filepath=filename)
    return None
