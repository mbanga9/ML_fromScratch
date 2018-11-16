import numpy as np

test = np.array([[1,2],[3,4]])
print(test)
test1 = np.repeat(test,2)
print(test1)
print(test1.reshape((test.shape[0],test.shape[1]*2)))