#!/usr/bin/env python3
"""This module contains a function that creates a batch normalization layer for
a neural network in tensorflow"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """This function creates a batch normalization layer for a neural network
in tensorflow"""

    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    lay = tf.layers.Dense(units=n, kernel_initializer=init)
    mean, var = tf.nn.moments(lay(prev), axes=[0])
    gamma = tf.get_variable(name="gamma", shape=[n],
                            initializer=tf.ones_initializer(), trainable=True)
    beta = tf.get_variable(name="beta", shape=[n],
                           initializer=tf.zeros_initializer(), trainable=True)
    out = tf.nn.batch_normalization(lay(prev), mean=mean, variance=var,
                                    offset=beta, scale=gamma,
                                    variance_epsilon=1e-8)
    return activation(out)
