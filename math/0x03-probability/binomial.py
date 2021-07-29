#!/usr/bin/env python3
"""This module contains the model for the binomial class"""


def factorial_calc(x):
    """This function calculates the factorial of a number"""
    if x == 0 or x == 1:
        return 1
    return x * factorial_calc(x - 1)


class Binomial:
    """This class represents a binomial distribution"""
    def __init__(self, data=None, n=1, p=0.5):
        """This function initializes the data members of the Binomial class"""
        if data is None:
            if n < 1:
                raise ValueError("n must be a positive value")
            elif p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            else:
                self.n = int(n)
                self.p = float(p)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                n = len(data)
                mu = sum(data) / n
                var = sum([(i - mu) ** 2 for i in data]) / n
                self.p = 1 - var / mu
                self.n = int(round(mu / self.p))
                self.p = float(mu / self.n)
