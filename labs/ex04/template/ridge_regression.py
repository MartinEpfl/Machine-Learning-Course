# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np


def ridge_regression(y, tx, lambda_):
    #Using L2
    lambdaPrime = lambda_*(2*tx.shape[0])
    x = tx
    gramLambda = x.T@x + np.identity(x.shape[1])*lambdaPrime

    w = np.linalg.solve(gramLambda,x.T@y)
    return w

