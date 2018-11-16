# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np
from costs import*

def least_squares(y, tx):
    
    w = np.linalg.solve(np.transpose(tx) @ tx,np.transpose(tx) @ y)
    #MSE = compute_loss(y,tx,w)
    return w