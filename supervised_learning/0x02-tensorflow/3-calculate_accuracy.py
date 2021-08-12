#!/usr/bin/env python3
"""This module contains a function that calculates the accuracy of a
prediction"""

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """This function calculates the accuracy of a prediction"""
    labs = tf.argmax(y, axis=1)
    preds = tf.argmax(y, axis=1)
    return tf.reduce_mean(tf.cast(tf.equal(labs, preds), dtype=tf.float32))
