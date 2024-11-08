# Packages
import random
import pandas as pd
import numpy as np
import time

# Custom Helpers
from src.activation_functions.activation_functions import relu
from src.helper_functions.helpers import train_test_split, mse


def predict(layers, nodes, swarm_size=3):
    # Data Pre-Processing
    data = pd.read_csv('/Users/camo/Library/Mobile Documents/com~apple~CloudDocs/MSc Courseworks/F21BC BIC - Coursework/Coursework/src/data/concrete_data.csv')

    # Inputs
    x = data[
        ['cement', 'blast_furnace_slag', 'fly_ash', 'water', 'superplasticizer', 'coarse_aggregate', 'fine_aggregate ',
         'age']].values.copy()
    dimensions = x.shape[1]

    # Output
    y = data[['concrete_compressive_strength']].values.copy()

    # Param Initialisation
    particles = []
    for i in range(swarm_size):
        particle = {
            "weights": np.random.uniform(-1, 1, (dimensions, nodes)).tolist(),
            "weights_hidden_layers": np.random.uniform(-1, 1, (dimensions, nodes)).tolist(),
            # Hard coding 1 for the number of output layers as we only want 1
            "weights_output": np.random.uniform(-1, 1, (nodes, 1)).tolist(),
            "biases": np.random.rand(nodes + 1).tolist(),
            "velocities": np.random.uniform(-1, 1, dimensions).tolist(),
            "bias_velocities": np.random.uniform(-1, 1, dimensions).tolist(),
            "fitness": 9999,
            "personal_best": [0],
            "personal_best_fitness": 9999,
            "informants": []
        }
        particles.append(particle)

    # Train, Test Split
    x_train, x_test = train_test_split(x)
    y_train, y_test = train_test_split(y)

    # Misc Params
    num_iterations = 10000
    start_time = time.time()

    # Predicted Output
    output = []

    for i in range(0, num_iterations):
        forward_pass_output = x_train

        for idx, particle in enumerate(particles):
            biases = np.array(particle['biases'])
            output = forward_pass(forward_pass_output,
                                               particle['weights_hidden_layers'],
                                               biases[:-1], particle['weights_output'],
                                               particle['biases'][-1])

            # For testing
            df = pd.DataFrame(output)
            df.to_csv('forward_pass_out.csv')

    result = mse(y_train, output)
    print('Training time: ', str(time.time() - start_time))
    print("Mean Squared Error (MSE):", result)


# Forward Pass
# Takes in Input Data, Weights and Biases
def forward_pass(data, weights, bias, weights_output, output_bias):
    ws = np.dot(data, weights) + bias
    a = relu(ws)

    return np.dot(a, weights_output) + output_bias