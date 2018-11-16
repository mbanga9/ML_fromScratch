#from proj1_helpers import *
from implementations_markus import *
from markus_plot import *
import numpy as np
import matplotlib.pyplot as plt
from tools import get_data
from proj1_helpers import *
#from implementations

def compute_gradient(y, tx, w,lambda_):
    """Compute the gradient."""
    f_p = y - (tx @ w)

    e = np.dot(np.transpose(tx),f_p)

    coeff = -1/((len(y)))
    result = (coeff)*e+lambda_*np.linalg.norm(w)
    return result

def compute_loss(y, tx, w,lambda_):
    #Return the mean square error
    coeff = 1/(2*(len(y)))
    mse = y - (tx @ w)
    result = coeff*(mse.T.dot(mse))+lambda_*np.linalg.norm(w)
    return result


def gradient_descent(y, tx, initial_w, max_iters, gamma, lambda_):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss

    name ="test.csv"



    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient(y, tx, w,lambda_)

        loss = compute_loss(y, tx, w,lambda_)

        w = w - (gamma*grad)
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws
lambda_ = 1
max_iters = 50
gamma = 0.1
sub_sample = True
y_train, x_train, ids_train, y_test_X, x_test_X, ids_test_X = get_data(sub_sample, large=False)
initial_w = np.random.random((x_train.shape[1],1))

gradient_descent(y_train, x_train, initial_w, max_iters, gamma,lambda_)
