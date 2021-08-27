#!/usr/bin/env python3
"""This module contains a function that creates a layer of a neural network
using dropout"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """This function creates a layer of a neural network using dropout"""
    weights = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    dropper = tf.layers.Dropout(rate=keep_prob)
    out = tf.layers.Dense(n, activation=activation, kernel_initializer=weights,
                          kernel_regularizer=dropper)
    return out(prev)
