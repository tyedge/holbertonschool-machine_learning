#!/usr/bin/env python3
"""This module contains the model for the DeepNeuralNetwork class"""

import numpy as np


class DeepNeuralNetwork:
    """This class defines a deep neural network performing binary
classification"""
    def __init__(self, nx, layers):
        """This function initializes the data members of the Neuron class"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) < 1:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(self.L):
            if type(layers[i]) is not int or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            if i == 0:
                self.__weights["W{}".format(i + 1)] = (np.random.randn(
                    layers[i], nx) * np.sqrt(2 / nx))
            else:
                self.__weights["W{}".format(i + 1)] = (np.random.randn(
                    layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1]))
            self.__weights["b{}".format(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """This function is the getter function for private attribute L"""
        return self.__L

    @property
    def cache(self):
        """This function is the getter function for private attribute cache"""
        return self.__cache

    @property
    def weights(self):
        """This function is the getter function for private attribute
weights"""
        return self.__weights
