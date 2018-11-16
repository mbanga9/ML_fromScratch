# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    x_mat = x[...,np.newaxis]
    new_x = np.repeat(x_mat,degree+1,1)
    power = np.arange(degree+1)
    new_x = np.power(new_x,power)
    return new_x
