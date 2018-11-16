import numpy as np
import  scipy.stats as stats
from implementations import *
from proj1_helpers import *
from cost import *
from gradients import *
from implementations import *
from tools import *


#Setting the seed for all the compitations
seed = np.random.seed(1)

#Loading the training Dataset
data_path = '/Users/youssefjanjar/Documents/GitHub/ML2017_GroupWork/Project/Data/train.csv'
y, input_data, ids = load_csv_data(data_path, sub_sample=False)



#Handeling the NaN value in the input data.
#input_data[input_data< -999.0] = np.nan
#np.nan_to_num(input_data)

col_mean = np.nanmedian(input_data,axis=0)


#Find indicies that you need to replace
inds = np.where(np.isnan(input_data))

#Place column means in the indices. Align the arrays using take
input_data[inds]=np.take(col_mean,inds[1])

print('Shape de input_data avant manips',input_data.shape)

print('Shape de input_data apres manips',input_data.shape)


input_data = np.delete(input_data, 22 , axis=1)

#Standardisation of our input data for the trainning
x, mean_x, std_x = standardize(input_data)


print('Shape de x apres manips',x.shape)

#Loading the testing Dataset for the submission
data_path2 = '/Users/youssefjanjar/Documents/GitHub/ML2017_GroupWork/Project/Data/test.csv'
y_test_sub ,input_data_test,ids_test = load_csv_data(data_path2, sub_sample=False)

input_data_test = np.delete(input_data_test, 22 , axis=1)
x_sub,mean_x_sub,std_x_sub = standardize(input_data_test)


#Script for the cross validation
'''
k_fold = 10
lambda_ = 1
k_indices = build_k_indices(y,k_fold,seed)
l = 1
M_rmse_tr = []
M_rmse_te = []
for k in range(k_fold):
            loss_tr,loss_te = cross_validation(y, x, k_indices, k, l)
            M_rmse_tr.append(loss_tr)
            M_rmse_te.append(loss_te)

print('Training loss:',M_rmse_tr)
print('Testting loss:',M_rmse_te)'''


#We set all the parameters that we need for our computation.
ratio = 0.8
degree = 7
max_iters = 100
gamma = 0.02
lambda_ = 0.001

#We split our training set 80% training and 20% for local validation.
x_train,x_test,y_train,y_test = split_data(x,y,ratio,seed)

#Building polynomials for the input data.
x = build_poly(x,degree)
x_train = build_poly(x_train,degree)

#We build the initial_w randomly according to the shape of our x_train
initial_w = np.random.random((x_train.shape[1],1))

#Training (Computation of the w)

#loss,w = reg_logistic_regression_SGD(y_train,x_train,lambda_,initial_w,max_iters,gamma)
#loss,w = logistic_regression_SGD(y_train,x_train,initial_w,max_iters,gamma)
#loss,w = least_squares_GD(y_train, x_train, initial_w, max_iters, gamma)
#loss,w = least_squares_SGD(y_train, x_train, initial_w, max_iters, gamma)
loss,w = ridge_regression(y_train,x_train, lambda_)
#loss,w = least_squares(y_train,x_train)



#Local testing on 20% of the training set as our local validation test set.
pred = x @ w
pred = np.sign(pred)
print( np.mean(pred.flatten() == y) )

#Creating the csv file for the submission on kaggle.
x_sub = build_poly(x_sub,degree)
pred_y = predict_labels(w,x_sub)
create_csv_submission(ids_test,pred_y,'ridge_poly.csv')