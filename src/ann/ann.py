import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.z_history = []
        self.weights = []
        self.biases = []
        previous_size = input_size
        for hidden_size in hidden_sizes:
            self.weights.append(np.random.randn(previous_size, hidden_size))
            self.biases.append(np.zeros((1,hidden_size)))
            previous_size = hidden_size
        self.weights.append(np.random.randn(previous_size, self.output_size))
        self.biases.append(np.zeros((1,self.output_size)))

    def activate(self, x):
        return x

    def forward_pass(self, x):
        activation = x
        self.z_history = []
        for i in range(len(self.weights) - 1):
            z = np.dot(activation, self.weights[i] + self.biases[i])
            self.z_history.append(z)
            activation = np.tanh(z)
        z = np.dot(activation, self.weights[-1] + self.biases[-1])
        self.z_history.append(z)
        output = self.activate(z)
        return output
    
if __name__=='__main__':
    neural_network = NeuralNetwork(2, [4,3], 1)
    print("Network output:", neural_network.forward_pass(np.array([[0.5,-1.2], [1.0, 0.8]])))
    #print(neural_network.z_history)
    #print("Network output:", neural_network.forward_pass(np.array([[0.1,-0.5], [1.5, 2.3], [-0.7, 0.4]])))
    #print(neural_network.z_history)
    num_layers = len(neural_network.z_history)
    fig, axes = plt.subplots(1, num_layers, figsize=(15, 5))

    for i, z_values in enumerate(neural_network.z_history):
        axes[i].plot(z_values.flatten(), marker='o')
        axes[i].set_title(f'Layer {i+1} Z-Values')
        axes[i].set_xlabel('Neuron Index')
        axes[i].set_ylabel('Z Value')
        axes[i].grid(True)
    plt.tight_layout()
    plt.show()

