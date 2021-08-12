#!/usr/bin/env python3
"""This module contains a function that calculates the softmax cross-entropy
loss of a prediction"""

import tensorflow as tf


def calculate_loss(y, y_pred):
    """This function calculates the softmax cross-entropy loss of a
prediction"""
    return tf.losses.softmax_cross_entropy(y, y_pred)
