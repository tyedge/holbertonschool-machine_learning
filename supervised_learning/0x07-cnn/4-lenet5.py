#!/usr/bin/env python3
"""This module contains a funtion that builds a modified version of the LeNet-5
architecture using tensorflow"""

import tensorflow as tf


def lenet5(x, y):
    """This function  builds a modified version of the LeNet-5 architecture
using tensorflow"""
    init = tf.contrib.layers.variance_scaling_initializer()
    relu = tf.nn.relu
    conv1 = tf.layers.Conv2D(filters=6, kernel_size=(5, 5), padding="same",
                             activation=relu, kernel_initializer=init)(x)
    pool1 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)
    conv2 = tf.layers.Conv2D(filters=16, kernel_size=(5, 5), padding="valid",
                             activation=relu, kernel_initializer=init)(pool1)
    pool2 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)
    flat = tf.layers.Flatten()(pool2)
    full1 = tf.layers.Dense(units=120, activation=relu,
                            kernel_initializer=init)(flat)
    full2 = tf.layers.Dense(units=120, activation=relu,
                            kernel_initializer=init)(full1)
    fullout = tf.layers.Dense(units=10, kernel_initializer=init)(full2)
    softmx = tf.nn.softmax(fullout)
    loss = tf.losses.softmax_cross_entropy(y, fullout)
    train = tf.train.AdamOptimizer().minimize(loss)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(fullout, 1),
                                               tf.argmax(y, 1)), tf.float32))
    return softmx, train, loss, accuracy
