import numpy as np

from src.helper_functions.helpers import forward_pass
from src.helper_functions.helpers import mse
import random


# Calculating the new velocity
# Used: https://stackoverflow.com/questions/15069998/particle-swarm-optimization-inertia-factor
# For guidance on the inertia constant
# Set to 0.8 for now, may investigate decreasing the inertia over time in future
def update_velocity(current_velocity,
                    cognitive_weight,
                    social_component,
                    particle_current_pos,
                    particle_best_pos,
                    particle_informant_best_pos,
                    best,
                    inertia=0.7):

    b = cognitive_weight
    c = social_component

    # Update the velocity using the PSO formula
    new_velocity = (inertia * current_velocity +
                    b * (particle_best_pos - particle_current_pos) +
                    c * (particle_informant_best_pos - particle_current_pos))

    # We want to use Numpy.Clip to ensure that none of the values
    # take us out of bounds (are too big)
    # https://numpy.org/doc/stable/reference/generated/numpy.clip.html
    new_velocity = np.clip(new_velocity, -0.08, 0.08)

    return new_velocity

# Mean Squared Error has been chosen
# for the fitness function
def calculate_fitness(y_pred, y_train, layers, particle):
    y_pred = forward_pass(y_pred, layers, particle)
    return mse(y_train, y_pred)

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
