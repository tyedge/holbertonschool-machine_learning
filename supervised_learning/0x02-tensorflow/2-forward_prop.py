#!/usr/bin/env python3
"""This module contains a function that creates the forward propagation graph
for the neural network"""

import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """This function creates the forward propagation graph
for the neural network"""
    pred = x
    for i in range(len(layer_sizes)):
        pred = create_layer(pred, layer_sizes[i], activations[i])
    return pred
