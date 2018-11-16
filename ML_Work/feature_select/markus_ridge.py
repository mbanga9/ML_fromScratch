from proj1_helpers import *
from implementations import *
from markus_plot import *
import numpy as np
import matplotlib.pyplot as plt

#from implementations

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

sub_sample = True
y_train, x_train, ids_train, y_test, x_test, ids_test = get_data(sub_sample, large=False)
name ="test.csv"

print(np.shape(x_train))



def ridge_regression_demo(y_train, x_train, y_test, x_test):
    """ridge regression demo."""
    # define parameter
    lambdas = np.logspace(-5, 1, 15)
    rmse_tr = []
    rmse_te = []
    y_train, mean_y, std_y = standardize(y_train)
    x_train, mean_xt, std_xt = standardize(x_train)
    x_test, mean_xp, std_xp = standardize(x_test)
    #print(np.shape(mean_xp), np.shape(std_xp))
    #exit()
    for ind, lambda_ in enumerate(lambdas):



        w, MSE_train = ridge_regression(y_train, x_train, lambda_)
        #w,_ = ridge_regression_markus(y_train, x_train, lambda_)

        RMSE_train = compute_mse_markus(y_train, x_train, w, mean_y, std_y)
        RMSE_test = compute_mse_markus(y_test, x_test, w, mean_y, std_y)
        rmse_tr.append(RMSE_train)
        rmse_te.append(RMSE_test)
        print("lambda={l:.3f}, Training RMSE={tr:.3f}, Testing RMSE={te:.3f}".format(
               l=lambda_, tr=rmse_tr[ind], te=rmse_te[ind]))
    return w


weights = ridge_regression_demo(y_train, x_train, y_test, x_test)

#y_pred = predict_labels(weights, x_test)

#create_csv_submission(ids_test, y_pred, name)


"""
weights = ridge_regression_demo(y_train, x_train, y_test, x_test)

# from stochastic_gradient_descent import *

# Define the parameters of the algorithm.
max_iters = 30
gamma = 10**-5

# Initialization
w_initial = np.array([0, 0])

# Start SGD.
print(weights)
sgd_losses, sgd_ws = least_squares_GD(
    y_train, x_train, weights, max_iters, gamma)
print(sgd_ws)

RMSE_test = compute_mse(y_test, x_test, sgd_ws)

# Print result
print("Final RMSE", RMSE_test)
"""
