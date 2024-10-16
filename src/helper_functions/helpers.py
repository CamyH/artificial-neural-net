# Mean Squared Error (MSE) chosen for regression

import numpy as np

# n = data points
# y = observed values
# yhat = predicted values
def mse(n, y, yhat):
    if n == 0:
        raise Exception('data points cannot be zero for MSE')

    return 1 / n * np.sum((yhat - y) ** 2)

# Split the data into train and test data
def train_test_split(arr, percent=0.2):
    split_idx = int(len(arr) * percent)
    arr_train = arr[:split_idx]
    arr_test = arr[split_idx:]

    return arr_train, arr_test
