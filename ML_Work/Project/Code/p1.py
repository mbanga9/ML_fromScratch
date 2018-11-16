from implementations import *
from proj1_helpers import *
import numpy as np
from cost import *
from gradients import *
from implementations import *
from tools import *
from lasso import lasso
"""
data_path = "/Users/markusekvall/Desktop/ML_course/projects/ML2017_GroupWork/data/train_small.csv"
yb, input_data, ids = load_csv_data(data_path, sub_sample=False)

data_path2 = "/Users/markusekvall/Desktop/ML_course/projects/ML2017_GroupWork/data/test_small.csv"
y_test_sub ,input_data_test,ids_test = load_csv_data(data_path2, sub_sample=False)
"""
#Loading the training Dataset
data_path = "/Users/markusekvall/Desktop/ML_course/projects/ML2017_GroupWork/Project/Code/Data/train.csv"
yb, input_data, ids = load_csv_data(data_path, sub_sample=True)

data_path2 = "/Users/markusekvall/Desktop/ML_course/projects/ML2017_GroupWork/Project/Code/Data/test.csv"
y_test_sub ,input_data_test,ids_test = load_csv_data(data_path2, sub_sample=True)


#index =[0,5,7,11,13,14,17,18,20]
#index1 = [ 0,  1,  3,  5,  7, 11, 13, 15, 17, 18, 20]
#index = [ 0,  1,  5,  7, 10, 11, 13, 14, 15, 17, 18, 19, 20, 22]

#input_data = input_data[:,index]
#input_data_test = input_data_test[:,index]




seed = np.random.seed(1)
x, mean_x, std_x = standardize(input_data)
y = yb

x_sub,mean_x_sub,std_x_sub = standardize(input_data_test)

#Initialization of the wights w
"W[250 000X1]"

seed = np.random.seed(1)

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

ratio = 0.8
degree = 7
x_train,x_test,y_train,y_test = split_data(x,y,ratio,seed)


max_iters = 30
gamma = 0.01
lambda_ = 0.01
"""
lambdas = np.logspace(-15, 1, 15)

for lambda_ in lambdas:
    print(lambda_)
    index = lasso(x_train,y_train,lambda_)
    print(index)
print("hej")
for lambda_ in np.linspace(0,10,15):
    print(lambda_)
    index = lasso(x_train,y_train,lambda_)
    print(index)

print("hej")
for lambda_ in np.logspace(1,10,15):
    print(lambda_)
    index = lasso(x_train,y_train,lambda_)
    print(index)
exit()
index = lasso(x_train,y_train,lambda_)
exit()
x_train = build_poly(x_train,degree)
initial_w = np.random.random((x_train.shape[1],1))

"""
x_train = build_poly(x_train,degree)
initial_w = np.random.random((x_train.shape[1],1))

print('x_train',x_train.shape)
print('np.shape(x_train)',np.shape(x_train))
loss_train,w = reg_logistic_regression_SGD(y_train,x_train,lambda_,initial_w,max_iters,gamma)
#loss_train, w =  logistic_regression_SGD(y_train, x_train, initial_w ,max_iters ,gamma)
#loss_train, w = ridge_regression(y_train, x_train, lambda_)
#loss_test = calculate_loss(y_test,x_test,w)
#w = least_squares(y_train, x_train)
#loss, w = least_squares_GD(y_train, x_train, initial_w, max_iters, gamma)
#loss, w = least_squares_SGD(y_train, x_train, initial_w, max_iters, gamma)
x = build_poly(x ,degree)
#print(np.power(x[0][30],2))
#print(np.power(x[0][30],2))
#print(x[0][59])
pred = x @ w
pred = np.sign(pred)
print(pred.shape, y.shape)

print(pred[0:10], y[0:10])
print( np.mean(pred.flatten() == y) )

x_sub = build_poly(x_sub,degree)
pred_y = predict_labels(w,x_sub)
create_csv_submission(ids_test,pred_y,'ridge_reg_lasso.csv')
