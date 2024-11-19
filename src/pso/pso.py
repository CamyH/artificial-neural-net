import numpy as np
from src.ann.neuralnetwork import NeuralNetwork

def particle_swarm_optimisation(num_parameters, ann_architecture, ann_activations, data, labels, swarm_size=50, c1=1.5, c2=1.5):

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
            fitness = calculate_fitness(data, labels, particle[idx]['position'], ann_architecture, ann_activations)

            if fitness < particle[idx]['best_fitness']:
                particle[idx]['best_position'] = particle[idx]['position'].copy()
                particle[idx]['best_fitness'] = fitness

            if fitness < global_best_fitness:
                global_best_position = particle[idx]['position'].copy()
                global_best_fitness = fitness

        for idx, particle in enumerate(swarm):
            updated_position = update_particle(particle[idx]['velocity'], particle[idx]['position'], particle[idx]['best_position'], c1, c2, global_best_position)
            particle[idx]['position'] = updated_position

        print(f"Iteration {iteration + 1}/{max_iter}, Fitness: {global_best_fitness}")

    return global_best_position



def calculate_fitness(data, labels, particle_position, ann_architecture, ann_activations):
    # Update the ANN with the new parameters from PSO
    ann = NeuralNetwork(ann_architecture, ann_activations)
    ann.update_parameters(particle_position)

    # Forward pass to generate predictions
    predictions = []
    for item in data:
        predictions.append(ann.forward_pass(item.reshape(-1, 1)))
    predictions = np.array(predictions).flatten()
    return np.mean(np.abs(predictions - labels.flatten()))

def update_particle(current_velocity, particle_current_position, particle_best_position, global_best_position, c1, c2):
    r1, r2 = np.random.rand(), np.random.rand()
    inertia = 0.5
    cognitive_component = c1 * r1 * (particle_best_position - particle_current_position)
    social_component = c2 * r2 * (global_best_position - particle_current_position)

    new_velocity = inertia * current_velocity + cognitive_component + social_component
    particle_new_position = particle_current_position + new_velocity

    return particle_new_position