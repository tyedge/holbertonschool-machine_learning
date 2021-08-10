#!/usr/bin/env python3
"""This module contains the model for the neuron class"""

import numpy as np
import matplotlib.pyplot as plt


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

    def forward_prop(self, X):
        """This method calculates the forward propagation of the neuron"""
        out = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-out))
        return self.__A

    def cost(self, Y, A):
        """This method calculates cost of model using logistic regression"""
        cost = -(1/Y.size) * np.sum(Y*np.log(A) + ((1-Y)*np.log(1.0000001-A)))
        return cost

    def evaluate(self, X, Y):
        """This method evaluates the neuron’s predictions"""
        predi = np.where(self.forward_prop(X) >= 0.5, 1, 0)
        cost = self.cost(Y, self.forward_prop(X))
        return predi, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """This method calculates one pass of gradient descent on the neuron"""
        self.__W -= alpha * (np.matmul(X, (A - Y).T) / Y.size).T
        self.__b -= alpha * (np.sum(A - Y) / Y.size)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """This method trains the neuron"""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        cost = []
        rations = []
        for i in range(iterations + 1):
            '''self.gradient_descent(X, Y, self.forward_prop(X), alpha)'''
            if i % step == 0 or i == iterations:
                rations.append(i)
                cost.append(self.cost(Y, self.forward_prop(X)))
                if verbose:
                    print(
                        "Cost after {} iterations: {}".
                        format(i, self.cost(Y, self.forward_prop(X))))
            self.gradient_descent(X, Y, self.forward_prop(X), alpha)
        if graph:
            plt.plot(rations, cost)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()
        return self.evaluate(X, Y)
