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

    def forward_prop(self, X):
        """This method calculates the forward propagation of the neural
network"""
        self.__cache["A0"] = X
        for i in range(self.__L):
            out = np.matmul(self.__weights["W{}".format(i + 1)], self.__cache[
                "A{}".format(i)]) + self.__weights["b{}".format(i + 1)]
            self.__cache["A{}".format(i + 1)] = 1 / (1 + np.exp(-out))
        return self.__cache["A{}".format(i + 1)], self.cache

    def cost(self, Y, A):
        """This method calculates the cost of the model using logistic
regression"""
        return -(1 / Y.size) * np.sum(Y * np.log(A) + ((1 - Y) * np.log(
            1.0000001 - A)))

    def evaluate(self, X, Y):
        """This method evaluates the neural network’s predictions"""
        self.forward_prop(X)
        predi = np.where(self.__cache["A{}".format(self.__L)] >= 0.5, 1, 0)
        cost = self.cost(Y, self.__cache["A{}".format(self.__L)])
        return predi, cost
