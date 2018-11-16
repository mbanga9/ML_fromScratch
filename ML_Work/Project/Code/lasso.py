import numpy as np
from implementations import *
from proj1_helpers import *
from cost import *
from gradients import *
from implementations import *
from tools import *

"""
data_path = "/Users/markusekvall/Desktop/ML_course/projects/ML2017_GroupWork/data/train_small.csv"
yb, input_data, ids = load_csv_data(data_path, sub_sample=False)

data_path2 = "/Users/markusekvall/Desktop/ML_course/projects/ML2017_GroupWork/data/test_small.csv"
y_test_sub ,input_data_test,ids_test = load_csv_data(data_path2, sub_sample=False)

#Loading the training Dataset
data_path = "/Users/markusekvall/Desktop/ML_course/projects/ML2017_GroupWork/Project/Code/Data/train.csv"
yb, input_data, ids = load_csv_data(data_path, sub_sample=False)

data_path2 = "/Users/markusekvall/Desktop/ML_course/projects/ML2017_GroupWork/Project/Code/Data/test.csv"
y_test_sub ,input_data_test,ids_test = load_csv_data(data_path2, sub_sample=False)


x, mean_x, std_x = standardize(input_data)
"""

def lasso(x,y,lambda_):
    y = y - np.mean(y)
    #step 1 pick residual and eps
    r = y
    #w = np.zeros(len(r))
    #Step 2 find highest correlated feature
    [n,d] = np.shape(x)
    #print(np.shape(d))
    w = np.zeros(d)
    #Corr needs to be reseted every looå
    for i in range(0, 2000):
        corr = 0
        for feat in range(0, d):
            c = np.correlate(r,x[:,feat])
            if c > corr:
                corr = c
                high_feat = feat
                # Step 3 extend the weight
        delta = lambda_*np.sign(np.dot(x[:,high_feat].T, r))
        w[high_feat] = w[high_feat] + delta
        # Step 4
        r = r -delta*x[:,high_feat]
    a = np.nonzero(w)
    return a[0]
