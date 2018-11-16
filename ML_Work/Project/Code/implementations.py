import numpy as np
from proj1_helpers import *
from cost import *
from gradients import *
from implementations import *
from tools import *



def compute_mse(y, tx, w):
    #Return the mean square error
    coeff = 1/(2*(len(y)))
    mse = y - (tx @ w)
    result = coeff*(mse.T.dot(mse))
    return result

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

def least_squares_SGD(y, tx, initial_w,max_iters, gamma):
               
    loss = float('inf')
    ws = [initial_w]
    losses = []
    w = initial_w
    batch_size = 256

    for n_iter in range(max_iters):
        for Y, tX in batch_iter(y, tx, batch_size):
               
                gradient = compute_gradient_mse(Y, tX, w)
                #print('Gradient:',gradient)
                loss = compute_mse(Y, tX, w)
                w = w - (gamma*gradient)
                #print('w',w)
                ws.append(w)
                losses.append(loss)  
               
    #We only keep track of the last values of the w and the loss            
    return loss, w

def least_squares(y, tx):
    w = np.linalg.solve(np.transpose(tx) @ tx, np.transpose(tx) @ y)
    loss = compute_mse(y, tx, w)
    return loss,w

def ridge_regression(y, tx, lambda_):         
    [n,d] = np.shape(tx)
    XtX =np.dot(tx.T,tx)
    inv_XtX = np.linalg.inv(XtX + lambda_*np.identity(d)*2*n)
    w = np.dot(inv_XtX, (np.dot(tx.T, y)))
    mse = compute_mse(y, tx, w) + lambda_*(w.T.dot(w))          
    return mse,w
         

def penalized_logistic_regression(y, tx, w, lambda_):
    loss = calculate_loss(y, tx, w) + lambda_ * np.squeeze(w.T@w)
    grad = compute_gradient_likelihood(y, tx, w) +  2 * lambda_ * w
    return loss, grad

def logistic_regression_SGD(y, tx, initial_w,max_iters, gamma):
               
    loss = float('inf')
    ws = [initial_w]
    losses = []
    w = initial_w
    batch_size = 128
               
    for n_iter in range(max_iters):
        for Y, tX in batch_iter(y, tx, batch_size):
                grad = compute_gradient_likelihood(Y, tX, w)
                loss = calculate_loss(Y, tX, w)
                w = w - (gamma*grad)
                ws.append(w)
                losses.append(loss)    
    #We only keep track of the last values of the w and the loss            
    return loss, w

               
def reg_logistic_regression_SGD(y, tx, lambda_, initial_w, max_iters, gamma):
    #Regularized logistic regression with regularisation factor lambda using gradient descent.           
    w = initial_w
    losses = []
    threshold = 1e-8
    batch_size = 256
               
    for i in range(max_iters):
        for Y, tX in batch_iter(y, tx, batch_size):

            loss,grad = penalized_logistic_regression(Y,tX,w,lambda_)
            losses.append(loss)
            
            w = w - gamma*grad
            
            #print(np.linalg.norm(w))

    return loss, w


               
               