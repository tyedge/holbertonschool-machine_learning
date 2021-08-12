#!/usr/bin/env python3
"""This module contains a function that returns two placeholders"""

import tensorflow as tf


def create_placeholders(nx, classes):
    """This function returns placeholders, x and y, for the neural network"""
    x = tf.placeholder(tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(tf.float32, shape=(None, classes), name='y')
    return x, y
