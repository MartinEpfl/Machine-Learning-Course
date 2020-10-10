# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    packed =  np.vstack([y,x]).T
    np.random.shuffle(packed)
    N = y.shape[0]
    eightyN = int(ratio*N)
    xTrain = packed[0:eightyN,1]
    yTrain = packed[0:eightyN,0]
    xTest = packed[eightyN:N, 1]
    yTest = packed[eightyN:N,0]
    # ***************************************************
    # INSERT YOUR CODE HERE
    # split the data based on the given ratio: TODO
    # ***************************************************
    return xTrain, yTrain, xTest, yTest