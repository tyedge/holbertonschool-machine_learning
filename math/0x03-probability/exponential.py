#!/usr/bin/env python3
"""This module contains the model for the exponential class"""


e = 2.7182818285


class Exponential:
    """This class represents an exponential distribution"""
    def __init__(self, data=None, lambtha=1.):
        """This function initializes the Exponential class"""
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
                self.lambtha = float(len(data) / sum(data))

    def pdf(self, x):
        """Calculates the value of the PDF for a given time period"""

        lamb = self.lambtha
        ex_factor = e**-lamb*x

        if x < 0:
            return 0
        else:
            return lamb * ex_factor

    def cdf(self, x):
        """Calculates the value of the CDF for a given time period"""

        lamb = self.lambtha
        ex_factor = e**-lamb*x

        if x < 0:
            return 0
        else:
            return 1 - ex_factor
