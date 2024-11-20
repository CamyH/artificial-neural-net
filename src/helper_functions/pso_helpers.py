# Packages
import numpy as np
import random

# Custom Hooks
from ann.neural_net import setup, predict
from helper_functions.helpers import mse

# Function to calculate the new velocity of a particle
# Also updates the particles position with the new velocity
# Used: https://stackoverflow.com/questions/15069998/particle-swarm-optimization-inertia-factor
# For guidance on the inertia constant
def update_particle(current_velocity,
                    particle_current_position,
                    particle_best_position,
                    global_best_position,
                    c1,
                    c2,
                    c3,
                    best_informant,
                    jump_size=0.5,
                    inertia=0.5):
    r1, r2 = np.random.rand(), np.random.rand()
    cognitive_component = c1 * r1 * (particle_best_position - particle_current_position)
    social_component = c2 * r2 * (global_best_position - particle_current_position)
    informant_component = c3 * (best_informant - particle_current_position)

    new_velocity = inertia * current_velocity + cognitive_component + social_component + informant_component

    # Add step to velocity to help exploration
    new_velocity = new_velocity + jump_size * (best_informant - particle_current_position)

    # Boundary handling
    new_velocity = np.clip(new_velocity, -0.6, 0.6)

    updated_position = particle_current_position + new_velocity

    return updated_position

# Mean Squared Error has been chosen
# for the fitness function
def calculate_fitness(layer_dimensions,
                      data,
                      labels,
                      particle_position):
    # Initialise the weights and baises for the neural net
    weights, biases = setup(layer_dimensions)

    # Forward pass to generate predictions
    predictions = []
    for item in data:
        predictions.append(predict(item.reshape(-1, 1), weights, biases, particle_position))
    predictions = np.array(predictions).flatten()
    return mse(labels, predictions)

# Randomly assign n (informants_size) number of informants per particle
# Default is 2, can be overridden
# Returns the indices of the informant particle for each particle
def init_informants(num_particles, informants_size=2):
    informants = []
    for _ in range(num_particles):
        informant = np.random.choice(num_particles, informants_size, replace=False)
        informants.append(informant)

    return informants
