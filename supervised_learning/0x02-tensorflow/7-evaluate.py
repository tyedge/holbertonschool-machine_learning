#!/usr/bin/env python3
"""This module contains a function that evaluates the output of a neural
network"""

import tensorflow as tf


def evaluate(X, Y, save_path):
    """This function evaluates the output of a neural network"""
    sp = save_path
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(sp + ".meta")
        saver.restore(sess, sp)

        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]

        pred = tf.get_collection("y_pred")[0]
        loss = tf.get_collection("loss")[0]
        acc = tf.get_collection("accuracy")[0]

    return sess.run(pred, {x: X, y: Y}), sess.run(
        accuracy, {x: X, y: Y}), sess.run(loss, {x: X, y: Y})
