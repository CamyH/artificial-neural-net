# Packages
import numpy as np

# Custom Hooks
from helper_functions.pso_helpers import calculate_fitness, update_particle

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
        swarm.append(
            {i: {
                'position': position,
                'velocity': velocity,
                'best_position': best_position,
                'best_fitness': best_fitness,
                'informants': []
                }
            }
        )

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
