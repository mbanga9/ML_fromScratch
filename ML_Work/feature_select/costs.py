import numpy as np
from proj1_helpers import *


from implementations_markus import *
from tools import *

def compute_mse(y, tx, w):
    #Return the mean square error
    coeff = 1/(2*(len(y)))
    mse = y - (tx @ w)
    result = coeff*(mse.T.dot(mse))
    return result

def compute_loss_absolut(y, tx, w):
    #Return the absolute error
    coeff = 1/((len(y)))
    first_part = tx @ w
    scnd_part = y - first_part
    main = np.absolute(scnd_part)
    result = coeff*(main.sum(axis = 0))
    return result


def calculate_loss(y, tx, w):
    #Return the loss for the loglikehood ratio

    first = np.log(1 + np.exp(tx@w))
    second = y*(tx @ w)
    result = (first-second).sum(axis = 0)
    return result
