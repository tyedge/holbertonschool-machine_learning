#!/usr/bin/env python3
"""This module contains two functions: one that saves a model’s configuration
in JSON format and another which loads a model with a specific configuration"""


import tensorflow.keras as K


def save_config(network, filename):
    """This function saves a model’s configuration in JSON format"""
    with open(filename, 'w') as file:
        file.write(network.to_json())
    return None


def load_config(filename):
    """This function loads a model with a specific configuration"""
    with open(filename, "r") as file:
        return K.models.model_from_json(file.read())
