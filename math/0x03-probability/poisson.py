#!/usr/bin/env python3
"""This module contains the model for the poisson class"""


def factorial_calc(x):
    """This function calculates the factorial of a number"""
    if x == 0 or x == 1:
        return 1
    return x * factorial_calc(x - 1)


e = 2.7182818285


class Poisson:
    """This class represents a poisson distribution"""
    def __init__(self, data=None, lambtha=1.):
        """This function initializes the data members of the Poisson class"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of successes"""
        if k < 0:
            return 0
        if type(k) is not int:
            k = int(k)
        mu = self.lambtha
        return ((e**-mu) * (mu**k)) / factorial_calc(k)

    def cdf(self, k):
        """Calculates the value of the CDF for a given number of successes"""
        if k < 0:
            return 0
        if type(k) is not int:
            k = int(k)
        ret = 0
        for i in range(k + 1):
            ret += self.pmf(i)
        return ret
