import random
import numpy as np

from src.helper_functions.helpers import mse


# Calculating the new velocity
def update_velocity(current_velocity,
                    cognitive_weight,
                    social_component,
                    particle_current_pos,
                    particle_best_pos,
                    particle_informant_best_pos,
                    inertia=1,
                    beta=1,
                    gamma=1,
                    delta=1):

    b = random.uniform(0.0, beta)
    c = random.uniform(0.0, gamma)
    d = random.uniform(0.0, delta)

    new_velocity = (inertia * current_velocity +
                       b * (particle_best_pos - particle_current_pos) +
                       c * (particle_informant_best_pos - particle_current_pos))

    return new_velocity[0]

# Mean Squared Error has been chosen
# for the fitness function
def calculate_fitness(data_points, target_vals, true_targets):
    return mse(data_points, target_vals, true_targets)

