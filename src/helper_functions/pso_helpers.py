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
                    inertia=0.8):

    b = random.uniform(0.0, cognitive_weight)
    c = random.uniform(0.0, social_component)

    # Update the velocity using the PSO formula
    new_velocity = (inertia * current_velocity +
                    b * (particle_best_pos - particle_current_pos) +
                    c * (particle_informant_best_pos - particle_current_pos))

    return new_velocity

# Mean Squared Error has been chosen
# for the fitness function
def calculate_fitness(y_pred, y_train):
    return mse(y_train, y_pred)

# Randomly assign n number of informants
# Default is 2, can be overridden
def init_informants(particles, informants_size=2):
    # Using random.sample to return a new list of size
    # informants_size of the particles to be used as
    # the informants for each particle
    for particle in particles:
        # Filter out the current particle so we do not
        # assign a particle to be its own informant
        filtered_particles = particles.copy()
        filtered_particles.remove(particle)
        chosen_particles = random.sample(filtered_particles, informants_size)
        particle['informants'] = chosen_particles

    return particles
