# Packages
import random
import pandas as pd
import numpy as np
import time

# Custom Helpers
from src.activation_functions.activation_functions import relu, gradient_descent, update_weight_loss
from src.helper_functions.helpers import train_test_split, mse, mae
from src.helper_functions.pso_helpers import init_informants
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
    max_weight = 0.3
    min_weight = -0.3
    for j in range(swarm_size):
        particle = {
            'weights': [],
            'weights_output': np.random.uniform(min_weight, max_weight, (nodes, 1)),
            'biases': [],
            'velocities': [],
            'bias_velocities': np.random.uniform(-0.1, 0.1, dimensions),
            'fitness': 9999,
            'personal_best': [],
            'personal_best_fitness': 9999,
            'informants': []
        }

        prev_layer_nodes = dimensions
        for i in range(layers):
            weights = np.random.uniform(min_weight, max_weight, (prev_layer_nodes, nodes))
            biases = np.full(nodes, 0.1)
            velocities = np.random.uniform(-0.3, 0.3, (prev_layer_nodes, nodes))
            personal_best = np.zeros((prev_layer_nodes, nodes))

            #print(f'Layer {i}, weights shape {weights.shape} and velocities shape {velocities.shape}')

            particle['weights'].append(weights)
            particle['biases'].append(biases)
            particle['velocities'].append(velocities)
            particle['personal_best'].append(personal_best)

            prev_layer_nodes = nodes

        particles.append(particle)

    # Initialise Informants
    particles = init_informants(particles)

    # Train, Test Split
    x_train, x_test = train_test_split(x)
    y_train, y_test = train_test_split(y)

    # Misc Params
    num_iterations = 10000
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
            pso_output = particle_swarm_optimisation(forward_pass_output,
                                                    output,
                                                    dimensions,
                                                    particles)
            #print('weights ', particle['weights'])
            #print('pso_output ', pso_output['weights'])
            particle['weights'] = pso_output['weights']
            # For testing
            #df = pd.DataFrame(output)
            #df.to_csv('forward_pass_out.csv')

    #print(output.shape)
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