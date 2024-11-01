import random

import pandas as pd
import numpy as np
import time

from src.activation_functions.activation_functions import relu
from src.helper_functions.helpers import train_test_split, mse
from src.helper_functions.pso_helpers import forward_pass
from src.pso.pso import particle_swarm_optimisation
from src.pso.pso_2 import particle_swarm_optimisation_2


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
    iterations = 0
    particles = []
    for i in range(swarm_size):
        particle = {
            "positions": np.random.uniform(-1, 1, (swarm_size, dimensions)).tolist(),
            "biases": np.random.rand(dimensions).tolist(),
            "velocities": np.random.uniform(-1, 1, dimensions).tolist(),
            "bias_velocities": np.random.uniform(-1, 1, dimensions).tolist(),
            "fitness": 9999,
            "personal_best": [0],
            "personal_best_fitness": 9999,
            "informants": []
        }

        particles.append(particle)

    print(particles)
    #particles = np.random.rand(swarm_size, dimensions)

    # Train, Test Split
    x_train, x_test = train_test_split(x)
    y_train, y_test = train_test_split(y)


    # Min/Max Scaling
    min_val = np.min(y_train)
    max_val = np.max(y_train)

    #y_train_test = (y_train - min_val) / (max_val - min_val)
    #df = pd.DataFrame(x_train)
    #df.to_csv('test.csv')

    #df = pd.DataFrame(y_train)
    #df.to_csv('test_y.csv')

    x_train = np.array(x_train)

    #print('x_train', x_train)

    num_iterations = 10000
    start_time = time.time()



    for i in range(0, num_iterations):
        '''weights_biases = particle_swarm_optimisation_2(x_train,
                                                       y_train,
                                                       swarm_size,
                                                       dimensions,
                                                       particles)
'''
    forward_pass_output = x_train
    for idx, particle in enumerate(particles):
        forward_pass_output = forward_pass(forward_pass_output, particle['positions'], particle['biases'][idx])
        print('forward pass output ', forward_pass_output)
