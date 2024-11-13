
from src.helper_functions.pso_helpers import update_velocity, calculate_fitness
import numpy as np

def particle_swarm_optimisation(y_pred,
                                y_train,
                                dimensions,
                                particles,
                                layers):
    iterations = 0
    cognitive_weight = 1.1
    # The absolute best position & fitness seen by any particle
    best = {
        'weights': [],
        'fitness': 0
    }
    while iterations < 10:
        for idx, particle in enumerate(particles):
            for j, weight in enumerate(particle['weights']):
                #print('pb ', particle['personal_best_fitness'])
                fitness = calculate_fitness(y_pred, y_train, layers, particle)
                #print('fitness ', fitness)
                if best['fitness'] == 0 or fitness < best['fitness']:
                    best['weights'] = particle['weights']
                    best['fitness'] = fitness
                    particle['personal_best'][j] = particle['weights']
                    particle['personal_best_fitness'] = fitness
                for dimension in range(dimensions):
                    updated_velocity = update_velocity(
                                       particle['velocities'][j],
                                       cognitive_weight,
                                       weight[j],
                                       weight,
                                       particle['personal_best'][j][j],
                                       particle['personal_best'][j][j],
                                       best['weights'])
                    #print(updated_velocity[0], updated_velocity[1])

                    particle['velocities'][j] += updated_velocity
                    particle['weights'][j] += updated_velocity
                    #print('BLAGH ', particle['weights'][idx][dimension])
                    #particle['weights'][idx][dimension] = np.clip(particle['weights'][idx][dimension], -1, 1)
                    #print('hi ', particle['weights'])


        #print(best['weights'], best['fitness'])
        #print(iterations)
        iterations += 1

    return best