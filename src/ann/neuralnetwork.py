import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes, activation_functions):
        """
        Initialize the ANN with specified layers and activation functions.
        Parameters:
        - layer_sizes: List containing the number of nodes in each layer.
        - activation_functions: List of activation functions for each layer.
        """
        self.layer_sizes = layer_sizes
        self.activation_functions = activation_functions
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i-1]) * np.sqrt(2 / layer_sizes[i-1]) 
                        for i in range(1, len(layer_sizes))]
        self.biases = [np.random.randn(layer_sizes[i], 1) * 0.1 for i in range(1, len(layer_sizes))]

    def activate(self, x, func):
        if func == 'linear':
            return x
        elif func == 'logistic':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif func == 'relu':
            return np.maximum(0, x)
        elif func == 'tanh':
            return np.tanh(x)
        else:
            raise ValueError("Unknown activation function.")

    # Forward Pass
    # Takes in the set of inputs, returns the predicted outputs
    def forward_pass(self, data):
        output = data
        for i in range(len(self.weights) - 1):
            ws = np.dot(self.weights[i], output) + self.biases[i]
            output = self.activate(ws, self.activation_functions[i])

        # Don't want to apply any activation function on the output
        return np.dot(self.weights[-1], output) + self.biases[-1]

    # Update the params of the network with new ones
    def update_parameters(self, params):
        idx = 0
        for i in range(len(self.weights)):
            weight_shape = self.weights[i].shape
            bias_shape = self.biases[i].shape
            self.weights[i] = params[idx:idx + np.prod(weight_shape)].reshape(weight_shape)
            idx += np.prod(weight_shape)
            self.biases[i] = params[idx:idx + np.prod(bias_shape)].reshape(bias_shape)
            idx += np.prod(bias_shape)
