#!/usr/bin/env python3
"""This module contains two functions: one that saves an entire model and
another which loads an entire model"""


import tensorflow.keras as K


def save_model(network, filename):
    """This function saves an entire model"""
    network.save(filepath=filename)
    return None


def load_model(filename):
    """This function loads an entire model"""
    return K.models.load_model(filepath=filename)
