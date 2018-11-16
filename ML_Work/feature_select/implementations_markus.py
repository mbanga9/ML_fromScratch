import numpy as np
#from hw_helpers import *

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    #x = np.linalg.solve(x, std_x)
    return x, mean_x, std_x

def de_standardize(x, mean_x, std_x):
    """Reverse the procedure of standardization."""
    x = x * std_x
    x = x + mean_x
    return x

def compute_mse_markus(y, tx, w, m, s):
    #Return the mean square error
    coeff = 1/(2*(len(y)))
    #mse = y - tx @ w * s + m
    mse = y - (tx @ w)*s + m
    result = coeff*(mse.T.dot(mse))
    return result

def sigmoid(t):
    """apply sigmoid function on t."""
    smd = np.exp(t)/(1+np.exp(t))
    return smd

def compute_mse(y, tx, w):
    #Return the mean square error
    coeff = 1/(2*(len(y)))
    mse = y - tx @ w
    result = coeff*(mse.T.dot(mse))
    return result

def compute_loss_absolut(y, tx, w):
    #Return the absolute error
    coeff = 1/((len(y)))
    first_part = tx @ w
    scnd_part = y - first_part
    main = np.absolute(scnd_part)
    result = coeff*(main.sum(axis=0))
    return result


def calculate_loss(y, tx, w):
    #Return the loss for the loglikehood ratio
    first = np.log(1 + np.exp(tx@w))
    second = y*(tx@ w)
    result = (first-second).sum(axis = 0)
    return result


def compute_gradient_mse(y, tx, w):
    #Computes the gradient for least_sqaure
    first = y-(tx @ w)
    coeff = -1/(y.shape[0])
    result = (coeff)*(tx.T.dot(first))
    return result


def compute_gradient_likelihood(y, tx, w):
    #Computes the gradient for the likelihood ratio
    first = sigmoid(tx.dot(w))- y
    result = tx.T @ first
    return result

def calculate_gradient_penalized(y, tx, w,lambda_):
   #Computes the gradient for the penalized likelihood ratio
    first = sigmoid(tx.dot(w))- y
    result = (tx.T @ first)-(lambda_*(w.T.dot(w)))
    return result

"""
def calculate_hessian(y, tx, w):
    #Computes the hessian Matrix
    S = np.diag((sigmoid(tx@w)*(1-sigmoid(tx@w))).T[0])
    H = tx.T @ S @ tx
    return H
"""
def calculate_hessian(y, tx, w):
    """return the hessian of the loss function."""

    [n,d] = np.shape(tx)

    sig = sigmoid(np.dot(tx, w))
    S = sig*(1-sig)*np.identity(n)
    return np.dot(tx.T, np.dot(S, tx))



def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient_mse(y, tx, w)
        loss = compute_mse(y, tx, w)
        w = w - (gamma*grad)
        ws.append(w)
        losses.append(loss)
    #We only keep track of the last values of the w and the loss
    return loss, w

def least_squares_SDG(y, tx, initial_w,max_iters, gamma):

    loss = float('inf')
    ws = [initial_w]
    losses = []
    w = initial_w
    batch_size = 1

    for n_iter in range(max_iters):
        for Y, tX in batch_iter(y, tx, batch_size):

                gradient = compute_gradient_mse(Y, tX, w)
                #We compute the SDG with the mean_square_error
                loss = compute_loss(Y, tX, w)
                w = w - (gamma*gradient)
                ws.append(w)
                losses.append(loss)
    #We only keep track of the last values of the w and the loss
    return losse, w

def least_squares(y, tx):
    w = np.linalg.solve(np.transpose(tx) @ tx, np.transpose(tx) @ y)
    #w = np.inv(np.dot(tx.T,tx))
    return w

def ridge_regression(y, tx, lambda_):

    [n,d] = np.shape(tx)
    XtX =np.dot(tx.T,tx)
    #inv_XtX = np.linalg.inv(XtX + lambda_*np.identity(d)*2*n)
    #w = np.dot(inv_XtX, (np.dot(tx.T, y)))
    inv_XtX = np.linalg.inv(XtX + lambda_*np.identity(d)*2*n)
    w = np.linalg.solve(XtX + lambda_*np.identity(d)*2*n, (np.dot(tx.T, y)))
    mse = compute_mse(y, tx, w)
    return w,mse

def ridge_regression_markus(y, tx, lambda_):
    [n,d] = np.shape(tx)

    XtX = np.dot(tx.T, tx)
    V, W2, Vh = np.linalg.svd(XtX)
    #R2 = lambda_*np.identity(d)*2*n
    R2 = 2*n*np.ones(len(W2))*lambda_
    inv_W2_reg = (W2 + R2)**-1
    XtX_reg_inv = np.dot(np.dot(Vh.T, np.diag(inv_W2_reg)), Vh)
    coefs = np.dot(XtX_reg_inv, np.dot(tx.T, y.T))
    Neff = np.sum(W2*inv_W2_reg)
    return coefs, Neff


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    #Logistic regression using gradient descent.
    w = initial_w
    losses = []
    threshold = 1e-8

    for i in range(max_iters):
        grad = compute_gradient_likelihood(y, tx, w)
        w = w-(gamma*grad)
        loss = calculate_loss(y, tx, w)
        losses.append(loss)

        if len(losses) > 1 and np.abs(losses[-1] - losses[-2] < threshold):
            break
    loss = calculate_loss(y, tx, w)
    return loss, w


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    #Regularized logistic regression with regularisation factor lambda using gradient descent.
    w = initial_w
    losses = []
    threshold = 1e-8

    for i in range(max_iters):
        loss = calculate_loss(y, tx, w) + (lambda_/2)*(w@w.T)
        losses.append(loss)
        g = compute_gradient_penalized(y, tx, w)
        w = w - gamma*g
        if len(losses) > 1 and np.abs(losses[-1]- losses[-2] < threshold):
            break

    return loss, w

def newton_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    #Logistic regression using Newton's method.
    losses = []
    w = initial_w
    threshold = 1e-8

    for i in range(max_iters):
        loss = calculate_loss(y, tx, w)
        losses.append(loss)
        g = compute_gradient_likelihood(y, tx, w)
        H = calculate_hessian(y, tx, w)
        w = w-(gamma*(np.linalg.inv(H) @ g))
        if len(losses) > 1 and np.abs(losses[-1]- losses[-2] < threshold):
            break
    #We calculate the final loss with the current w.
    loss = calculate_loss(y, tx, w)
    return loss, w
