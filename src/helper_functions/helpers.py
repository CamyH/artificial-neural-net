# Packages
import numpy as np

# Custom Hooks
from activation_functions.activation_functions import relu, tan, log

# Mean Squared Error (MSE) chosen for regression
# y = observed values
# pred = predicted values/yhat
def mse(y, pred):
    return np.mean((pred - y) ** 2)

# Mean Absolute Error
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Split the data into train and test data
def train_test_split(arr, percent=0.2):
    split_idx = int(len(arr) * (1 - percent))
    arr_train = arr[:split_idx]
    arr_test = arr[split_idx:]

    return arr_train, arr_test

# Calculate and return the total number of parameters
# in the neural network
def params_count(weights, biases):
    # Iterate through each of the weights and biases and calculate the total
    count = 0
    for weight_matrix in weights:
        count += np.prod(weight_matrix.shape)
    for bias_vector in biases:
        count += np.prod(bias_vector.shape)
    return count

# Function to return the correct activation function based on
# what the user specifies
# uses the new-ish match case statement
# https://www.geeksforgeeks.org/python-match-case-statement/
# Returns RELU by default
def activation_function(specified_activation_function, ws):
    match specified_activation_function:
        case 'relu':
            return relu(ws)
        case 'tan':
            return tan(ws)
        case 'log':
            return log(ws)
        case _:
            return relu(ws)
