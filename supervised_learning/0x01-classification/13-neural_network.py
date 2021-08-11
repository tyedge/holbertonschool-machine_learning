#!/usr/bin/env python3
"""This module contains the model for the NeuralNetwork class"""

import numpy as np


class NeuralNetwork:
    """This class defines a single neuron performing binary classification"""
    def __init__(self, nx, nodes):
        """This function initializes the data members of the NeuralNetwork
class"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros(shape=(nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """This function is the getter function for private attribute W1"""
        return self.__W1

    @property
    def b1(self):
        """This function is the getter function for private attribute b1"""
        return self.__b1

    @property
    def A1(self):
        """This function is the getter function for private attribute A1"""
        return self.__A1

    @property
    def W2(self):
        """This function is the getter function for private attribute W2"""
        return self.__W2

    @property
    def b2(self):
        """This function is the getter function for private attribute b2"""
        return self.__b2

    @property
    def A2(self):
        """This function is the getter function for private attribute A2"""
        return self.__A2

    def forward_prop(self, X):
        """This method calculates the forward propagation of the neuron"""
        hide = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-hide))

        out = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-out))

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """This method calculates the cost of the model using logistic
regression"""
        cost = -(1/Y.size) * np.sum(Y*np.log(A) + ((1-Y)*np.log(1.0000001-A)))
        return cost

    def evaluate(self, X, Y):
        """This method evaluates the neural network’s predictions"""
        self.forward_prop(X)
        predi = np.where(self.__A2 >= 0.5, 1, 0)
        cost = self.cost(Y, self.__A2)
        return predi, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """This module calculates one pass of gradient descent on the neural
network"""
        out = np.matmul(self.W2.T, (A2 - Y)) * (A1 * (1 - A1))
        self.__W1 -= alpha * (np.matmul(out, X.T) / Y.size)
        self.__b1 -= alpha * (np.sum(out, axis=1, keepdims=True) / Y.size)
        self.__W2 -= alpha * (np.matmul((A2 - Y), A1.T) / Y.size)
        self.__b2 -= alpha * (np.sum((A2 - Y), axis=1, keepdims=True) / Y.size)
