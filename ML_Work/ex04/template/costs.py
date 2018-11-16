# -*- coding: utf-8 -*-
"""A function to compute the cost."""
import numpy as np

def compute_loss(y, tx, w):
    """compute the loss by mse."""
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse

def compute_RMSE(y, tx, w):
    return np.sqrt(2*compute_loss(y, tx, w))