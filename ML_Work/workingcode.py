import numpy as np
from costs import *

def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    coeff = 1/(2*(len(y)))
    first_part = tx @ w
    scnd_part = y - first_part
    main = np.power(scnd_part, 2)
    result = coeff*np.sum(main, axis=0)
    return result

def compute_loss_absolut(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    coeff = 1/((len(y)))
    first_part = tx @ w
    scnd_part = y - first_part
    main = np.absolute(scnd_part)
    result = coeff*np.sum(main,axis=0)
    return result


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    f_p = y-(tx @ w)
    e = np.dot(np.transpose(tx), f_p)

    coeff = -1/((len(y)))
    result = (coeff)*e
    return result

def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        w = w - (gamma*grad)
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws

def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        for Y, tX in batch_iter(y, tx, batch_size):
                gradient = compute_gradient(Y, tX, w)
                loss = compute_loss(Y, tX, w)
                w = w - (gamma*gradient)
                # store w and loss
                ws.append(w)
                losses.append(loss)
                print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                      bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws

def least_squares(y, tx):
    w = np.linalg.solve(np.transpose(tx) @ tx, np.transpose(tx) @ y)
    return w

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    [n,d] = np.shape(tx)
    XtX =np.dot(tx.T,tx)
    inv_XtX = np.linalg.inv(XtX + lambda_*np.identity(d)*2*n)
    w = np.dot(inv_XtX, (np.dot(tx.T, y)))
    return w
