import numpy as np

# Custom Imports
from src.activation_functions.activation_functions import relu


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

# Initialise weights and biases on a per layer
# bases for the neural network
def init_weights_biases(layers, weights, biases):
    for i in range(0, layers):
        weights.append(np.random.rand(weights[i], weights[i + 1]))
        biases.append(np.zeros(weights[i + 1]))

    return weights, biases

# Forward Pass
# Takes in Input Data, Weights and Biases
def forward_pass(data, layers, weights, biases):
    output = data
    # Don't want to iterate over the output layer so -1
    for i in range(layers - 1):
        weights = weights[i]
        bias = biases[i]
        ws = np.dot(output, weights) + bias
        output = relu(ws)

    return np.dot(output, weights[-1]) + biases[-1]

