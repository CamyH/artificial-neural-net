# Packages
import random
import pandas as pd
import numpy as np
import time

# Custom Helpers
from src.activation_functions.activation_functions import relu
from src.helper_functions.helpers import train_test_split, mse, mae, init_weights_biases
from src.pso.pso import particle_swarm_optimisation


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

    weights = []
    biases = []

    weights, biases = init_weights_biases(layers, weights, biases)

    # Train, Test Split
    x_train, x_test = train_test_split(x)
    y_train, y_test = train_test_split(y)

    # Misc Params
    num_iterations = 1000
    start_time = time.time()
    np.random.seed(1)

    # Predicted Output
    output = []

    for i in range(0, num_iterations):
        forward_pass_output = x_train

        for idx, particle in enumerate(particles):
            output = forward_pass(forward_pass_output,
                                               layers,
                                  particle)

            # PSO
            pso_output = particle_swarm_optimisation(swarm_size,
                                                    forward_pass_output,
                                                    output,
                                                    dimensions,
                                                    particles,
                                                     layers,
                                                     nodes)

            particle['weights'] = pso_output['weights']
            # For testing
            #df = pd.DataFrame(output)
            #df.to_csv('forward_pass_out.csv')

    result = mse(y_train, output)
    result_mae = mae(y_train, output)
    print('Training time: ', str(time.time() - start_time))
    print('Mean Squared Error (MSE):', result)
    print('Mean Absolute Error (MAE):', result_mae)


# Forward Pass
# Takes in Input Data, Weights and Biases
def forward_pass(data, layers, particle):
    output = data
    for i in range(layers):
        weights = particle['weights'][i]
        bias = particle['biases'][i]
        ws = np.dot(output, weights) + bias
        output = relu(ws)

    return np.dot(output, particle['weights_output']) + particle['biases'][-1][layers]

