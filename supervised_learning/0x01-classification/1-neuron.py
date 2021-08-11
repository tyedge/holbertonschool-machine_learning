#!/usr/bin/env python3
"""This module contains the model for the neuron class"""

import numpy as np


class Neuron:
    """This class defines a single neuron performing binary classification"""
    def __init__(self, nx):
        """This function initializes the data members of the Neuron class"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """This function is the getter function for private attribute W"""
        return self.__W

    @property
    def b(self):
        """This function is the getter function for private attribute b"""
        return self.__b

    @property
    def A(self):
        """This function is the getter function for private attribute A"""
        return self.__A
