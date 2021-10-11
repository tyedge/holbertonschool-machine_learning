#!/usr/bin/env python3
"""This module contains a function that updates the learning rate using
inverse time decay in numpy"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """This function updates the learning rate using inverse time decay in
numpy"""
    return alpha / (1 + decay_rate * int(global_step / decay_step))
