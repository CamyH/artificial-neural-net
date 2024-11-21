# Packages
import numpy as np

# Custom Hooks
from helper_functions.pso_helpers import calculate_fitness, update_particle, init_informants


def particle_swarm_optimisation(layer_dimensions,
                                num_parameters,
                                data, labels,
                                c1,
                                c2,
                                activation_function,
                                swarm_size=50,
                                c3=0.4):
    swarm = []
    informants = []
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
                'best_informant': [],
                'informants': []
                }
            }
        )

    informants = init_informants(swarm_size)

    for idx, particle in enumerate(swarm):
        particle[idx]['informants'] = informants[idx]

    global_best_position = None
    global_best_fitness = 9999
    max_iter = 10
    for iteration in range(max_iter):
        for idx, particle in enumerate(swarm):
            fitness = calculate_fitness(layer_dimensions, data, labels, particle[idx]['position'], activation_function)
            if fitness < particle[idx]['best_fitness']:
                particle[idx]['best_position'] = particle[idx]['position'].copy()
                particle[idx]['best_fitness'] = fitness

            if fitness < global_best_fitness:
                global_best_position = particle[idx]['position'].copy()
                global_best_fitness = fitness

        for idx, particle in enumerate(swarm):
            best_informant = None
            best_informant_fitness = 9999

            for j in particle[idx]['informants']:
                informant = swarm[j]
                informant_fitness = calculate_fitness(layer_dimensions, data, labels, informant[j]['position'], activation_function)

                if informant_fitness < best_informant_fitness:
                    best_informant = informant[j]['position'].copy()
                    best_informant_fitness = informant_fitness

                # If none of the informants are good enough then
                # just set the best informant to the current informant
                if best_informant is not None:
                    particle[idx]['best_informant'] = best_informant
                else:
                    particle[idx]['best_informant'] = informant[j]['position'].copy()

            for dimension in range(len(particle[idx]['position'])):
                # Updating the velocity/position of each dimension of the particle
                updated_position = update_particle(
                    particle[idx]['velocity'][dimension],
                    particle[idx]['position'][dimension],
                    particle[idx]['best_position'][dimension],
                    c1,
                    c2,
                    c3,
                    global_best_position[dimension],
                    particle[idx]['best_informant'][dimension])
                # Update the position of the specific dimension
                particle[idx]['position'][dimension] = updated_position

        print(f"Iteration {iteration + 1}/{max_iter}, Fitness: {global_best_fitness}")

    return global_best_position
