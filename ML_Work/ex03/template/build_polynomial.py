# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    
    x_mat = x[...,np.newaxis]
    new_x = np.repeat(x_mat,degree+1,axis=1)
    power = np.arange(degree+1)
    new_x = np.power(new_x,power)
    return new_x
