#!/usr/bin/env python3
"""This module contains a function that creates the training operation for a
neural network in tensorflow using the Adam optimization algorithm"""

import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """This function creates the training operation for a neural network in
tensorflow using the Adam optimization algorithm"""
    return tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon).minimize(loss)
