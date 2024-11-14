
from src.helper_functions.pso_helpers import update_velocity, calculate_fitness, init_informants
import numpy as np

def particle_swarm_optimisation(swarm_size,
                                y_pred,
                                y_train,
                                dimensions,
                                particles,
                                layers,
                                nodes):
    iterations = 10
    cognitive_weight = 0.4
    social_weight = 0.6

    max_weight = 0.2
    min_weight = -0.2

    for j in range(swarm_size):
        particle = {
            'positions': [],
            'biases': [],
            'velocities': [],
            'bias_velocities': np.random.uniform(-0.1, 0.1, dimensions),
            'fitness': 9999,
            'personal_best': [],
            'personal_best_fitness': 9999,
            'informants': []
        }

        prev_layer_nodes = dimensions
        for i in range(layers):
            positions = np.random.uniform(min_weight, max_weight, (prev_layer_nodes, nodes))
            biases = np.full(nodes, 0.1)
            velocities = np.random.uniform(-0.2, 0.2, (prev_layer_nodes, nodes))
            personal_best = np.zeros((prev_layer_nodes, nodes))

            #print(f'Layer {i}, positions shape {positions.shape} and velocities shape {velocities.shape}')

            particle['positions'].append(positions)
            particle['biases'].append(biases)
            particle['velocities'].append(velocities)
            particle['personal_best'].append(personal_best)

            prev_layer_nodes = nodes

        particles.append(particle)

    # Initialise Informants
    particles = init_informants(particles)

    # The absolute best position & fitness seen by any particle
    best = {
        'positions': [],
        'fitness': 0
    }

    for i in range(iterations):
        for idx, particle in enumerate(particles):
            fitness = calculate_fitness(y_pred, y_train, layers, particle)
            print('f'
                  'Fitness ', fitness)
            if best['fitness'] == 0 or fitness < best['fitness']:
                best['positions'] = particle['positions']
                best['fitness'] = fitness
                particle['personal_best'] = particle['positions']
                particle['personal_best_fitness'] = fitness

            for dimension in range(dimensions):
                # Passing in the current velocity, cognitive/social weight, current position
                # the particles personal best position, informants best and local best
                updated_velocity = update_velocity(
                                    particle['velocities'],
                                    cognitive_weight,
                                    social_weight,
                                    particle['positions'],
                                    particle['personal_best'],
                                    particle['personal_best'],
                                    best['positions'])

                particle['velocities'] = updated_velocity
                particle['positions'] += updated_velocity

            iterations += 1
            print('Iteration ', iterations)

        return best