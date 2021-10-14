#!/usr/bin/env python3
"""This module contains the model for the DeepNeuralNetwork class"""

import numpy as np
import matplotlib.pyplot as plt
import pickle


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

    def gradient_descent(self, Y, cache, alpha=0.05):
        """This method calculates one pass of gradient descent on the neural
network"""
        diff = self.__cache["A{}".format(self.__L)] - Y
        for i in range(self.__L, 0, -1):
            a = "A{}".format(i - 1)
            w = "W{}".format(i)
            b = "b{}".format(i)
            dub = (1 / Y.size) * (np.matmul(diff, self.__cache[a].T))
            bee = (1 / Y.size) * (np.sum(diff, axis=1, keepdims=True))
            diff = np.matmul(self.__weights[w].T, diff) * (
                self.__cache[a] * (1 - self.__cache[a]))
            self.__weights[w] = self.__weights[w] - alpha * dub
            self.__weights[b] = self.__weights[b] - alpha * bee

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """This method trains the deep neural network by updating the private
attributes __weights and __cache"""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            if type(step) != int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        costs = []
        for i in range(iterations + 1):
            A, cost = self.evaluate(X, Y)

            if verbose and i % step == 0:
                print("Cost after {} iterations: {}".format(i, cost))
            costs.append(cost)
            self.gradient_descent(Y, self.__cache, alpha)

        if graph:
            plt.plot([*list(range(iterations)), iterations], costs)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        """This method saves the instance object to a file in pickle format"""
        if not filename.endswith(".pkl", -4):
            filename = filename + ".pkl"
        with open(filename, "wb") as file:
            pickle.dump(self, file)
            file.close()

    @staticmethod
    def load(filename):
        """This static method loads a pickled DeepNeuralNetwork object"""
        try:
            with open(filename, "rb") as file:
                lowed = pickle.load(file)
                file.close()
                return lowed
        except FileNotFoundError:
            return None
