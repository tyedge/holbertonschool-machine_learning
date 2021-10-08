#!/usr/bin/env python3
"""This module contains a function that converts a label vector into a one-hot
matrix"""

import tensorflow.keras as K


def one_hot(labels, classes=None):
    """This function converts a label vector into a one-hot matrix"""

    return K.utils.to_categorical(labels, classes)
