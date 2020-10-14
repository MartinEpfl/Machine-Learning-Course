# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    """calculate the least squares."""
    x = tx
    gram = x.T @ x
    w = np.linalg.solve(gram,x.T@y)
    e = y - x@w
    MSE = 1/(2*y.shape[0])*(e.T@e)
    return w, MSE
