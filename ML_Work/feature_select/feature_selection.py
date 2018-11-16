#from proj1_helpers import *
from implementations_markus import *
from markus_plot import *
import numpy as np
import matplotlib.pyplot as plt
from tools import get_data
from proj1_helpers import *





def feature_selection(x_train, y_train, x_test, y_test, lambda_):
    [n_train,d] = np.shape(x_train)
    [n_test,d] = np.shape(x_test)
    LOSS = 10**6
    X_train_best = np.c_[np.ones(n_train)]
    X_test_best = np.c_[np.ones(n_test)]
    new_feature = True
    weights = []
    indicies = []
    while new_feature:
        new_feature = False
        [n_train, d_train] = np.shape(x_train)
        [n_test, d_test] = np.shape(x_test)
        for i in range(0, d_train):
            tx_train = np.c_[X_train_best, x_train[:, i]]
            tx_test = np.c_[X_test_best, x_test[:, i]]
            w, error = ridge_regression_markus(y_train, tx_train, lambda_)
            loss = compute_mse(y_test, tx_test, w)
            if loss < LOSS:
                weights = w
                LOSS = loss
                index = i
                new_feature = True
        if new_feature:
            indicies.append(index)
            X_train_best = np.c_[X_train_best, x_train[:, index]]
            X_test_best = np.c_[X_test_best, x_test[:, index]]
            x_train = np.delete(x_train, index, 1)
            x_test = np.delete(x_test, index, 1)
    return X_train_best, X_test_best, weights, indicies
