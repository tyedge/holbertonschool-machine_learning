#!/usr/bin/env python3
"""This module contains a function that creates a tensorflow layer that
includes L2 regularization"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """This function creates a tensorflow layer that includes L2
regularization"""
    weight = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    l2reg = tf.contrib.layers.l2_regularizer(lambtha)
    out = tf.layers.Dense(n, activation=activation,
                          kernel_initializer=weight, kernel_regularizer=l2reg)
    return out(prev)
