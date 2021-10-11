#!/usr/bin/env python3
"""This module contains a function that updates a variable using the RMSProp
optimization algorithm"""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """This function updates a variable using the RMSProp optimization
algorithm"""
    new = beta2 * s + (1 - beta2) * (grad ** 2)
    varu = var - alpha * grad / ((new ** 0.5) + epsilon)
    return varu, new
