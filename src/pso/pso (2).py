import numpy as np
from ann import ANN

class Particle:
    def __init__(self, num_parameters):
        self.position = np.random.randn(num_parameters)
        self.velocity = np.random.randn(num_parameters) * 0.1
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')

class PSO:
    def __init__(self, num_particles, num_parameters, ann_architecture, ann_activations, data, labels, swarm_size=50, informants=3, c1=1.5, c2=1.5, max_iter=200):
        self.ann_architecture = ann_architecture
        self.ann_activations = ann_activations
        self.data = data
        self.labels = labels
        self.swarm = [Particle(num_parameters) for _ in range(swarm_size)]
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter

    def fitness(self, particle_position):
        ann = ANN(self.ann_architecture, self.ann_activations)
        ann.set_parameters(particle_position)
        
        predictions = np.array([ann.forward(x.reshape(-1, 1)) for x in self.data]).flatten()
        return np.mean(np.abs(predictions - self.labels.flatten()))  # MAE

    def update_particle(self, particle):
        r1, r2 = np.random.rand(), np.random.rand()
        cognitive_component = self.c1 * r1 * (particle.best_position - particle.position)
        social_component = self.c2 * r2 * (self.global_best_position - particle.position)
        particle.velocity = 0.5 * particle.velocity + cognitive_component + social_component
        particle.position += particle.velocity

    def optimize(self):
        for iteration in range(self.max_iter):
            for particle in self.swarm:
                fitness = self.fitness(particle.position)

                if fitness < particle.best_fitness:
                    particle.best_position = particle.position.copy()
                    particle.best_fitness = fitness

                if fitness < self.global_best_fitness:
                    self.global_best_position = particle.position.copy()
                    self.global_best_fitness = fitness

            for particle in self.swarm:
                self.update_particle(particle)

            print(f"Iteration {iteration+1}/{self.max_iter}, Global Best Fitness: {self.global_best_fitness}")
        
        return self.global_best_position
