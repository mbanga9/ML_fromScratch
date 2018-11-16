import numpy as np
from proj1_helpers import *
from cost import *
from gradients import *
from implementations import *
from tools import *


def compute_gradient_mse(y, tx, w):
    #Computes the gradient for least_sqaure
    first = y[...,np.newaxis]-(tx.dot(w))
    coeff = -1/(y.shape[0])
    result = (coeff)*(tx.T.dot(first))
    return result
               
               
def compute_gradient_likelihood(y, tx, w):
    #Computes the gradient for the likelihood ratio
    first = sigmoid(tx.dot(w)) - y[...,np.newaxis]
    result = tx.T @ first     
    return result
               
def calculate_gradient_penalized(y, tx, w,lambda_):
   #Computes the gradient for the penalized likelihood ratio
    first = sigmoid(tx.dot(w))- y[...,np.newaxis]
    result = (tx.T @ first)+(lambda_*(w.T @w))
    return result
               
               
def calculate_hessian(y, tx, w):
    #Computes the hessian Matrix           
    S = np.diag((sigmoid(tx@w)*(1-sigmoid(tx@w))).T[0])
    H = tx.T @ S @ tx 
    return H             
               
               