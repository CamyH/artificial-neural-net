# Mean Squared Error (MSE) chosen for regression

import numpy as np

# n = data points
# y = observed values
# yhat = predicted values
def mse(n, y, yhat):
    if n == 0:
        raise Exception('data points cannot be zero for MSE')

    return 1 / n * np.sum((yhat - y) ** 2)
