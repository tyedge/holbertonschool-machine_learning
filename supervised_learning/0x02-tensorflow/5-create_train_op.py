#!/usr/bin/env python3
"""This module contains a function that creates the training operation for the
network"""

import tensorflow as tf


def create_train_op(loss, alpha):
    """This function calculates the softmax cross-entropy loss of a
prediction"""
    return tf.train.GradientDescentOptimizer(alpha).minimize(loss)
