#!/usr/bin/env python3
"""This module contains a function that updates a variable using the gradient
descent with momentum optimization algorithm"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """THis function updates a variable using the gradient descent with
momentum optimization algorithm"""
    vret = beta1 * v + (1 - beta1) * grad
    varu = var - vret * alpha
    return varu, vret
