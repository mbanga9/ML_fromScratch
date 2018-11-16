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
    marker = int(y.shape[0]*ratio)
    ids = np.arange(y.shape[0])
    np.random.shuffle(ids)
    ids_train = ids[:marker]
    ids_test = ids[marker:]
    
    return x[ids_train],x[ids_test],y[ids_train],y[ids_test]

