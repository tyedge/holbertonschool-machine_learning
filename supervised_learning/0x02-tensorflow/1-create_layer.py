#!/usr/bin/env python3
"""This module contains a function that returns the tensor output of the
layer"""

import tensorflow as tf


def create_layer(prev, n, activation):
    """This function returns tensor output of the layer"""
    weight = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    out = tf.layers.Dense(n, activation=activation, kernel_initializer=weight,
                          name="layer")
    return out(prev)
