import numpy as np
from proj1_helpers import *
#from cost import *
#from gradients import *
#from implementations import *




def sigmoid(t):
    return 1/(1 + np.exp(-t))

def standardize(x):
    #Standardize the original data set.yy
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x

def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    marker = int(y.shape[0]*ratio)
    ids = np.arange(y.shape[0])
    np.random.shuffle(ids)
    ids_train = ids[:marker]
    ids_test = ids[marker:]

    return x[ids_train],x[ids_test],y[ids_train],y[ids_test]



def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)
    num_batches = data_size // batch_size

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]



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
    x_train , y_train = x[train_idxs],y[train_idxs]
    x_test , y_test = x[test_idxs],y[test_idxs]
    w = least_squares(y_train,x_train)
    loss_tr = compute_mse(y_train,x_train,w)
    loss_te = compute_mse(y_test,x_test,w)

    return loss_tr, loss_te

def get_data(sub_sample, large=True):
    if large:
        data_path_train ="/Users/markusekvall/Desktop/ML_course/projects/ML2017_GroupWork/data/train.csv"
        data_path_test ="/Users/markusekvall/Desktop/ML_course/projects/ML2017_GroupWork/data/test.csv"
    else:
        data_path_train ="/Users/markusekvall/Desktop/ML_course/projects/ML2017_GroupWork/data/train_small.csv"
        data_path_test ="/Users/markusekvall/Desktop/ML_course/projects/ML2017_GroupWork/data/test_small.csv"
    yb_train, input_data_train, ids_train = load_csv_data(data_path_train,sub_sample=sub_sample)
    yb_test, input_data_test, ids_test = load_csv_data(data_path_test,sub_sample=sub_sample)
    return yb_train, input_data_train, ids_train, yb_test, input_data_test, ids_test
