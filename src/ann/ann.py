import numpy as np

class ANN:
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
        """Applies the specified activation function."""
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

    def forward(self, inputs):
        """Forward pass through the network."""
        a = inputs
        for i in range(len(self.weights)):
            z = np.dot(self.weights[i], a) + self.biases[i]
            a = self.activate(z, self.activation_functions[i])
        return a

    def set_parameters(self, params):
        """Set weights and biases from a flat parameter vector."""
        idx = 0
        for i in range(len(self.weights)):
            weight_shape = self.weights[i].shape
            bias_shape = self.biases[i].shape
            self.weights[i] = params[idx:idx + np.prod(weight_shape)].reshape(weight_shape)
            idx += np.prod(weight_shape)
            self.biases[i] = params[idx:idx + np.prod(bias_shape)].reshape(bias_shape)
            idx += np.prod(bias_shape)
