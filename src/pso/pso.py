import numpy as np
import random

from src.helper_functions.pso_helpers import *


def particle_swarm_optimisation(data_points,
                                target_values,
                                swarm_size,
                                dimensions):
    iterations = 0

    particles = np.random.rand(swarm_size, dimensions)

    cognitive_weight = 1.2
    social_component = 1.4

    for i in range(swarm_size):
        particles = np.vstack([particles, np.hstack([np.random.rand(swarm_size, dimensions)])])

    personal_bests = np.copy(particles)

    velocities = np.random.uniform(-1, 1, (swarm_size, dimensions))

    # Initialised to the first personal best
    best = np.full(swarm_size, -np.inf)

    while iterations < 10:
        for idx, particle in enumerate(particles):
            fitness = calculate_fitness(data_points, target_values, true_target_values)
            print(fitness)

            if best[idx] == -np.inf or fitness > best[idx]:  # Change from 0 to -np.inf for initialization
                best[idx] = fitness
                personal_bests[idx] = particle.copy()

            for dimension in range(dimensions):
                beta = random.uniform(0.0, 5.0)
                gamma = random.uniform(0.0, 10)
                delta = random.uniform(0.0, 15)
                # Update velocity
                velocities[idx] = update_velocity(velocities[idx],
                                                  cognitive_weight,
                                                  social_component,
                                                  idx,
                                                  best[idx],
                                                  personal_bests[idx])

                particles[idx][dimension] += velocities[idx][dimension]
                #print(velocities[idx])
                #print('velocities at idx ', velocities[idx])
                #velocities[idx] = velocity

            idx += 1


