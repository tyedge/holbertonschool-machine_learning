#!/usr/bin/env python3
"""This module contains a function that evaluates the output of a neural
network"""

import tensorflow as tf


def evaluate(X, Y, save_path):
    """This function evaluates the output of a neural network"""
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_path + ".meta")
        saver.restore(sess, save_path)

        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]

        pred = tf.get_collection('y_pred')[0]
        loss = tf.get_collection('loss')[0]
        acc = tf.get_collection('accuracy')[0]

    return sess.run([pred, acc, loss], {x: X, y: Y})
