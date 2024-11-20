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
                    inertia=0.5):
    r1, r2 = np.random.rand(), np.random.rand()
    cognitive_component = c1 * r1 * (particle_best_position - particle_current_position)
    social_component = c2 * r2 * (global_best_position - particle_current_position)

    new_velocity = inertia * current_velocity + cognitive_component + social_component
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

# Randomly assign n number of informants
# Default is 2, can be overridden
def init_informants(particles, informants_size=2):
    # Using random.sample to return a new list of size
    # informants_size of the particles to be used as
    # the informants for each particle
    for particle in particles:
        # Create a filtered list of particles excluding the current particle
        filtered_particles = []
        for p in particles:
            if p is not particle:
                filtered_particles.append(p)

        # Select random informants
        chosen_particles = random.sample(filtered_particles, informants_size)

        # Assign chosen informants to the current particle
        particle['informants'] = chosen_particles

    return particles
