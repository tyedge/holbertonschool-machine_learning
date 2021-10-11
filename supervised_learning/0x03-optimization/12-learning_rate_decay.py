#!/usr/bin/env python3
"""This module contains a function that creates a learning rate decay
operation in tensorflow using inverse time decay"""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """This function creates a learning rate decay operation in tensorflow
using inverse time decay"""

    return tf.train.inverse_time_decay(alpha, global_step, decay_step,
                                       decay_rate, staircase=True)
