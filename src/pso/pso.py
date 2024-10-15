import numpy as np
import random

from src.helper_functions.pso_helpers import *


def particle_swarm_optimisation(swarm_size):
    iterations = 0

    particles = np.empty((0, swarm_size))

    cognitive_weight = 1.2
    social_component = 1.4

    for i in range(swarm_size):
        particles = np.vstack([particles, np.hstack([np.random.rand(swarm_size)])])

    #print(particles)

    personal_bests = np.copy(particles)

    velocities = np.random.uniform(-1, 1, size=swarm_size)
    velocity = 0

    #print('velocities ', velocities[1])

    # Initialised to the first personal best
    best = np.full(swarm_size, -np.inf)
    #print('best', best)

    idx = 0

    while iterations < 10:
        for idx, particle in enumerate(particles):
            fitness = calculate_fitness(particle)
            print(fitness)

            if best[idx] == 0 or fitness > best[idx]:
                best[idx] = fitness
                personal_bests[idx] = particle.copy()

            for dimension in particle:
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

                #print(velocities[idx])
                #print('velocities at idx ', velocities[idx])
                #velocities[idx] = velocity

            idx += 1


