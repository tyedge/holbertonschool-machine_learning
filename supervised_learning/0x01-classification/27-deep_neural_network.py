#!/usr/bin/env python3
""" Notes """

import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    def __init__(self, nx, layers):
        """ Notes """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) < 1:
            raise TypeError("layers must be a list of positive integers")
        self.nx = nx
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(self.__L):
            if type(layers[i]) is not int or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")
            w = "W{}".format(i + 1)
            b = "b{}".format(i + 1)
            if i == 0:
                self.__weights[w] = (np.random.randn(layers[i], self.nx)
                                     * np.sqrt(2 / self.nx))
            else:
                self.__weights[w] = (np.random.randn(layers[i], layers[i-1])
                                     * np.sqrt(2 / layers[i - 1]))
            self.__weights[b] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """ Notes """
        return self.__L

    @property
    def cache(self):
        """ Notes """
        return self.__cache

    @property
    def weights(self):
        """ Notes """
        return self.__weights

    def forward_prop(self, X):
        """ Notes """
        self.__cache['A0'] = X

        for i in range(self.__L):
            w = "W{}".format(i + 1)
            b = "b{}".format(i + 1)
            ap = "A{}".format(i)
            af = "A{}".format(i + 1)

            z = np.matmul(self.__weights[w], self.__cache[ap]) \
                + self.__weights[b]

            if i == self.__L - 1:
                self.__cache[af] = (np.exp(z)/np.sum(np.exp(z), axis=0,
                                                     keepdims=True))
            else:
                self.__cache[af] = 1 / (1 + np.exp(-z))
        return self.__cache[af], self.__cache

    def cost(self, Y, A):
        """ Notes """
        return (-1 / (Y.shape[1])) * np.sum(Y * np.log(A))

    def evaluate(self, X, Y):
        """ Notes """
        self.forward_prop(X)[0]
        a = "A{}".format(self.__L)
        macs = np.amax(self.__cache[a], axis=0)
        return (np.where(self.__cache[a] == macs, 1, 0),
                self.cost(Y, self.__cache[a]))

    def gradient_descent(self, Y, cache, alpha=0.05):
        """ Notes """
        sz = Y.shape[1]
        diff = self.__cache["A{}".format(self.__L)] - Y
        for i in range(self.__L, 0, -1):
            a = "A{}".format(i - 1)
            w = "W{}".format(i)
            b = "b{}".format(i)
            dub = (1 / sz) * np.matmul(diff, self.__cache[a].T)
            bee = (1 / sz) * np.sum(diff, axis=1, keepdims=True)
            diff = np.matmul(self.__weights[w].T, diff) * (
                self.cache[a] * (1 - self.cache[a]))
            self.__weights[w] = self.__weights[w] - alpha * dub
            self.__weights[b] = self.__weights[b] - alpha * bee

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """ Notes """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        cost = []
        iters = []
        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(Y, self.cache, alpha)
            if i % step == 0 or i == iterations:
                c = self.cost(Y, self.__cache["A{}".format(self.L)])
                cost.append(c)
                iters.append(i)
                if verbose is True:
                    print("Cost after {} iterations: {}".format(i, c))
        if graph is True:
            plt.plot(iters, cost)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        """ Notes """
        if not filename:
            return None
        if filename[-4:] != ".pkl":
            filename = filename + ".pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """ Notes """
        try:
            with open(filename, 'rb') as f:
                ret = pickle.load(f)
            return ret
        except FileNotFoundError:
            return None
