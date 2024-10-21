import random
import numpy as np

from src.helper_functions.helpers import mse


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
                    inertia=0.8):

    new_velocity = ((inertia * current_velocity) +
                    cognitive_weight *
                    (particle_best_pos - particle_current_pos) +
                    social_component * (particle_informant_best_pos - particle_best_pos))

    return new_velocity

# Mean Squared Error has been chosen
# for the fitness function
def calculate_fitness(data_points, target_vals, true_targets):
    return mse(data_points, target_vals, true_targets)

