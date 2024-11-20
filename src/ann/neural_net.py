# Packages
import numpy as np

# Custom Hooks
from activation_functions.activation_functions import relu, tan, log

def setup(layer_dimensions):
    # Initialise the weights and biases using random initialsation
    weights = []
    biases = []
    for i in range(1, len(layer_dimensions)):
        weight = np.random.randn(layer_dimensions[i], layer_dimensions[i - 1])
        weights.append(weight)
        bias = np.full((layer_dimensions[i], 1), 0.1)
        biases.append(bias)

    return weights, biases


def predict(data, weights, biases, optimised_params):
    # Update the weights/biases with the optimised weights from PSO
    weights, biases = update_weights_biases(weights, biases, optimised_params)
    # Run forward pass and return predictions
    return forward_pass(data, weights, biases)

# Forward Pass
# Takes in the set of inputs, returns the predicted outputs
def forward_pass(data, weights, biases):
    output = data
    for i in range(len(weights) - 1):
        ws = np.dot(weights[i], output) + biases[i]
        output = relu(ws)

    # Don't want to apply any activation function on the output
    return np.dot(weights[-1], output) + biases[-1]

# Update the params of the network with new ones
def update_weights_biases(weights, biases, optimised_params):
    idx = 0
    for i in range(len(weights)):
        weight_shape = weights[i].shape
        bias_shape = biases[i].shape
        # We need to get the correct portion of the optimised weights to apply to the current weight
        weight_slice = optimised_params[idx:idx + np.prod(weight_shape)]
        weights[i] = weight_slice.reshape(weight_shape)
        idx += np.prod(weight_shape)
        # We now need to d othe same for the biases
        bias_slice = optimised_params[idx:idx + np.prod(bias_shape)]
        biases[i] = bias_slice.reshape(bias_shape)
        idx += np.prod(bias_shape)

    return weights, biases
