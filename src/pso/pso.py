from src.helper_functions.pso_helpers import calculate_fitness, update_velocity
import numpy as np

def particle_swarm_optimisation(y_pred,
                                y_train,
                                dimensions,
                                particles):
    iterations = 0
    cognitive_weight = 1.1
    # The absolute best position & fitness seen by any particle
    best = {
        'weights': [],
        'fitness': 0
    }

    for idx, particle in enumerate(particles):
        for j, weight in enumerate(particle['weights']):
            #print('pb ', particle['personal_best_fitness'])
            fitness = calculate_fitness(y_pred, y_train)
            print('fitness ', fitness)
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
                print(updated_velocity[0], updated_velocity[1])

                learning_rate = 0.05
                updated_velocity = np.clip(updated_velocity, -0.5, 0.5)
                test = learning_rate * updated_velocity
                particle['velocities'][j] += test
                particle['weights'][j] += test
                #print('BLAGH ', particle['weights'][idx][dimension])
                #particle['weights'][idx][dimension] = np.clip(particle['weights'][idx][dimension], -1, 1)
                #print('hi ', particle['weights'])


    #print(best['weights'], best['fitness'])
    iterations += 1

    return best