#%%
import sys
sys.path.append('../')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(x):
    """ Sigmoid function.
    This function accepts any shape of np.ndarray object as input and perform sigmoid operation.
    """
    sig = 1 / (1 + np.exp(-x))
    sig = np.minimum(sig, 0.9999)  # Set upper bound
    sig = np.maximum(sig, 0.0001)
    return sig

def der_sigmoid(y):
    """ First derivative of Sigmoid function.
    The input to this function should be the value that output from sigmoid function.
    """
    return sigmoid(y) * (1 - sigmoid(y))

# %%
x_train = pd.read_csv("X_train")
y_train = pd.read_csv("Y_train")
y_train = y_train["label"]
# %%
import random
from math import log

dim = x_train.shape[1]
w = [random.randrange(1, 10, 1) for i in range(dim)]
b = 2

iteration = 500
N = x_train.shape[0]
learning_rate = 0.2

def cross_entropy(y_hat, y):
    if(y_hat == 0 ):
        y_hat += 1e-15
    if(y_hat == 1):
        y_hat -= 1e-15

    if y == 1:
        return -1*log(y_hat)
    else:
        return -1*log(1 - y_hat)
N = 100
for it in range(iteration):
    x_value = 0
    for i in range(N):
        arr = x_train.iloc[i].to_numpy()
        x_value = np.inner(x_train.iloc[i], w)
        x_value += b
        sig = sigmoid(x_value)
        for x_i in range(dim):
            w[x_i] = w[x_i] - learning_rate * (-1 * (y_train[i] - sig)) * arr[x_i]
        b = b - learning_rate*(-1 * (y_train[i] - sig))
    if((it%10) == 0):
        print("iter %d" %(it))
        test_value = 0  
        error = 0  
        for i in range(N):
            test_value = np.inner(x_train.iloc[i], w)
            test_value += b
            sig = sigmoid(test_value)
            error += cross_entropy(sig, y_train[i])
        print(error)

# %%
