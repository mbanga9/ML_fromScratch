
from implementations_markus import *
from markus_plot import *
import numpy as np
import matplotlib.pyplot as plt
from tools import get_data
from proj1_helpers import *
from feature_selection import feature_selection
from markus_plot import *

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, x, k_indices, k, lambda_):
    train_idxs = [n for (i, idxs) in enumerate(k_indices)
                  for n in idxs if i != k]
    test_idxs = k_indices[k]
    x_train , y_train = x[train_idxs], y[train_idxs]
    x_test , y_test = x[test_idxs], y[test_idxs]
    tx_train = np.c_[np.ones(len(y_train)), x_train]
    x_test = np.c_[np.ones(len(y_test)), x_test]

    x_train, x_test, w, indicies = feature_selection(x_train, y_train, x_test, y_test, lambda_)
    loss_tr = compute_mse(y_train, x_train, w)
    loss_te = compute_mse(y_test, x_test, w)

    return loss_tr, loss_te, w, indicies

def cross_validation_demo():
    sub_sample = True
    y_train, x_train, ids_train, y_test_X, x_test_X, ids_test_X = get_data(sub_sample, large=True)
    seed = 1
    k_fold = 4
    lambdas = np.logspace(-4, 1, 30)
    k_indices = build_k_indices(y_train, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    W = []
    I = []
    for l in lambdas:
        M_rmse_tr = []
        M_rmse_te = []
        weight = []
        index1 = []
        for k in range(k_fold):
            loss_tr,loss_te, w, indices = cross_validation(y_train, x_train, k_indices, k, l)
            weight.append(w)
            index1.extend(indices)
            M_rmse_tr.append(loss_tr)
            M_rmse_te.append(loss_te)
        W.append(weight[np.argmin(M_rmse_te)])
        I.extend(index1)
        rmse_tr.append(np.mean(M_rmse_tr))
        rmse_te.append(np.mean(M_rmse_te))
    plt.hist(I, bins=np.arange(min(I), max(I)+1))
    plt.title("Frequency diagram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()
    plt.savefig("freq")

    """
    exit()
    ind = np.argmin(rmse_te)

    #print(ind)
    #print(np.shape(x_test_X[:,indices]))

    print(lambdas[ind],W[ind])#,rmse_te(rmse_te))
    cross_validation_visualization(lambdas, rmse_tr, rmse_te)

    x_test_X = np.c_[np.ones(len(y_test_X)), x_test_X[:,I[ind]]]
    x_test_X = np.c_[np.ones(len(y_test_X)),  x_test_X]
    #print("hej")

    pred = x_test_X @ W[ind]
    pred = np.sign(pred)
    #print(pred.shape, y.shape)

    #print(pred[0:10], y[0:10])
    print( np.mean(pred.flatten() == y_test_X) )
    y_pred = predict_labels(W[ind], x_test_X)

    create_csv_submission(ids_test_X, y_test_X, name)
    #y_pred = predict_labels(weights, x_test)

    #create_csv_submission(ids_test, y_pred, name)

"""
cross_validation_demo()
