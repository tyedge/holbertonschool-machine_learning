#!/usr/bin/env python3
"""This module contains a function that calculates the cost of a neural network
with L2 regularization"""

import tensorflow as tf


def l2_reg_cost(cost):
    """This function calculates the cost of a neural network with L2
regularization"""
    return cost + tf.losses.get_regularization_losses()
