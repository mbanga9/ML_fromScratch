# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np
from costs import compute_loss

def ridge_regression(y, tx, lambda_):
    w = np.linalg.inv(tx.T.dot(tx) + lambda_ *        np.identity(tx.shape[1])).dot(tx.T).dot(y)
    MSE = compute_loss(y,tx,w)
    return w,MSE