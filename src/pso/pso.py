import numpy as np
import random

from src.helper_functions.pso_helpers import *


def particle_swarm_optimisation(target_values,
                                swarm_size,
                                dimensions,
                                particles):
    iterations = 0

    cognitive_weight = 1.2
    social_component = 1.4

    personal_bests = np.copy(particles)

    velocities = np.random.uniform(-1, 1, (swarm_size, dimensions))

    # Initialised to the first personal best
    best = np.full(swarm_size, -np.inf)

    while iterations < 10:
        for idx, particle in enumerate(particles):
            print(particle)
            fitness = calculate_fitness(particle, target_values, target_values)
            print(fitness)

            if best[idx] == -np.inf or fitness > best[idx]:
                best[idx] = fitness
                personal_bests[idx] = particle.copy()

            for dimension in range(dimensions):
                # Update velocity
                velocities[idx] = update_velocity(velocities[idx],
                                                  cognitive_weight,
                                                  social_component,
                                                  idx,
                                                  best[idx],
                                                  personal_bests[idx])

                particles[idx][dimension] += velocities[idx][dimension]

            idx += 1

    return particles

