# Calculating the new velocity
def update_velocity(current_velocity,
                    inertia,
                    cognitive_weight,
                    social_component,
                    particle_current_pos,
                    particle_best_pos,
                    particle_informant_best_pos):
    # Calculating the new velocity
    new_velocity = (inertia * current_velocity +
                    cognitive_weight(particle_best_pos - particle_current_pos) +
                    social_component(particle_informant_best_pos - particle_current_pos))

    return new_velocity

# Mean Squared Error has been chosen
# for the fitness function
def calculate_fitness(particle_best_pos,
                      particle_current_pos):
    # train the ANN here and then we evaluate the fitness using MSE
    return 'not implemented'

