import numpy as np
from ann.neural_net import setup, predict
from helper_functions.helpers import mse


def particle_swarm_optimisation(layer_dimensions,
                                num_parameters,
                                data, labels,
                                swarm_size=50,
                                c1=1.5,
                                c2=1.5):
    swarm = []
    for i in range(swarm_size):
        position = np.random.randn(num_parameters)
        velocity = np.random.randn(num_parameters) * 0.1
        best_position = position.copy()
        best_fitness = 9999
        swarm.append({i: {'position': position, 'velocity': velocity, 'best_position': best_position, 'best_fitness': best_fitness}})

    global_best_position = None
    global_best_fitness = 9999
    max_iter = 10
    for iteration in range(max_iter):
        for idx, particle in enumerate(swarm):
            fitness = calculate_fitness(layer_dimensions, data, labels, particle[idx]['position'])

            if fitness < particle[idx]['best_fitness']:
                particle[idx]['best_position'] = particle[idx]['position'].copy()
                particle[idx]['best_fitness'] = fitness

            if fitness < global_best_fitness:
                global_best_position = particle[idx]['position'].copy()
                global_best_fitness = fitness

        for idx, particle in enumerate(swarm):
            for dim in range(len(particle[idx]['position'])):
                # Updating the velocity/position of each dimension of the particle
                updated_position = update_particle(
                    particle[idx]['velocity'][dim],
                    particle[idx]['position'][dim],
                    particle[idx]['best_position'][dim],
                    c1,
                    c2,
                    global_best_position[dim])
                # Update the position of the specific dimension
                particle[idx]['position'][dim] = updated_position

        print(f"Iteration {iteration + 1}/{max_iter}, Fitness: {global_best_fitness}")

    return global_best_position



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

# Function to update the velocity of a particle
# Also updates the particles position with the new velocity
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