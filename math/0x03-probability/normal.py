#!/usr/bin/env python3
"""This module contains the model for the normal class"""


e = 2.7182818285
pi = 3.1415926536


def errrr(n):
    """This function is the Normal Distribution error function"""
    return ((2/(pi**0.5)) * (n - ((n**3)/3) + ((n**5)/10) - ((n**7)/42) +
                             ((n**9)/216)))


class Normal:
    """This class represents a normal distribution"""
    def __init__(self, data=None, mean=0., stddev=1.):
        """This function initializes the Normal class"""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            else:
                self.mean = float(mean)
                self.stddev = float(stddev)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                n = len(data)
                mu = sum(data) / n
                var = sum([(i - mu) ** 2 for i in data]) / n
                self.mean = float(mu)
                self.stddev = float(var ** 0.5)

    def z_score(self, x):
        """Calculates the z-score of a given x-value"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the x-value of a given z-score"""
        return z * self.stddev + self.mean

    def pdf(self, x):
        """Calculates the value of the PDF for a given x-value"""
        d = (((2 * pi) ** 0.5) * self.stddev)
        n = e ** ((((x - self.mean) / self.stddev) ** 2) * -0.5)
        return (1 / d) * n

    def cdf(self, x):
        """Calculates the value of the CDF for a given x-value"""
        a = (x - self.mean) / (self.stddev * (2 ** 0.5))
        return (1 + errrr(a)) * 0.5
