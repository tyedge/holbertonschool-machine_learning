#!/usr/bin/env python3
"""This module contains a function that creates the training operation for a
neural network in tensorflow using the RMSProp optimization algorithm"""

import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """This function creates the training operation for a neural network in
tensorflow using the RMSProp optimization algorithm"""

    return tf.train.RMSPropOptimizer(learning_rate=alpha, decay=beta2,
                                     epsilon=epsilon).minimize(loss)
