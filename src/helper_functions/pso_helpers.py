from numpy.ma.core import shape

from src.activation_functions.activation_functions import relu, tan
from src.helper_functions.helpers import mse
import numpy as np
import random
import pandas as pd


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
    '''print('current v ', current_velocity)
    print('cognitive weight ', cognitive_weight)
    print('social component ', social_component)
    print('particle_current_pos ', particle_current_pos)
    print('particle_best_pos ', particle_best_pos)
    print('particle_informant_best_pos ', particle_informant_best_pos)'''

    b = random.uniform(0.0, cognitive_weight)
    c = random.uniform(0.0, social_component)

    # Update the velocity using the PSO formula
    new_velocity = (inertia * current_velocity +
                    b * (particle_best_pos - particle_current_pos) +
                    c * (particle_informant_best_pos - particle_current_pos))

    #print('NEW VELOCITY ', new_velocity)

    return new_velocity

# Mean Squared Error has been chosen
# for the fitness function
def calculate_fitness(x_train, weights, bias, y_train):
    #output = forward_pass(x_train, weights, bias)
    #print('output ' ,output)

    #return mse(y_train, output)
    return 0
