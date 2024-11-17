import random
import numpy as np

# Parameters
swarmsize = 10
alpha = 0.8    
beta = 0.5     
gamma = 0.5    
delta = 0.5    
epsi = 0.1     
dimensions = 2
target_best = 1e-6 #0.01  

print("Starting PSO with parameters:")
print(f"Swarm size: {swarmsize}")
print(f"Dimensions: {dimensions}")
print(f"Target fitness: {target_best}")
print("------------------------")

class Particle:
    def __init__(self):
        self.velocities = []
        self.position = np.array([0] * dimensions)
        self.fitness = float('inf')
        self.best_position = None
        self.informants = []

def fitness_function(x: np.ndarray) -> float:
    return np.sum(x**2)

def AssessFitness(x: Particle) -> float:
    return fitness_function(x.position)

def pso(swarmsize, alpha, beta, gamma, delta, epsi, dimensions):
    print("Initializing particles...")
    P = []
    # Initialize particles
    for i in range(swarmsize):
        particle = Particle()
        particle.velocities = np.random.uniform(-0.1, 0.1, dimensions)
        particle.position = np.random.uniform(-10, 10, dimensions)
        particle.best_position = particle.position.copy()
        P.append(particle)
        print(f"Particle {i} initialized at position {particle.position}")
    
    print("\nSetting up ring topology...")
    # Set up ring topology
    for i in range(swarmsize):
        P[i].informants = [
            P[(i-1) % swarmsize],
            P[i],
            P[(i+1) % swarmsize]
        ]
    print("Ring topology setup complete")
    print("------------------------")

    Best = None
    iteration = 0
    max_iterations = 1000

    print("Starting main optimization loop...")
    while iteration < max_iterations:
        iteration += 1
        
        # Update personal bests and global best
        for x in P:
            current_fitness = AssessFitness(x)
            if Best is None or current_fitness < AssessFitness(Best):
                Best = x
                x.best_position = x.position.copy()
                print(f"\nNew best solution found!")
                print(f"Position: {Best.position}")
                print(f"Fitness: {AssessFitness(Best)}")

        # Update velocities and positions
        for x in P:
            best_informant = min(x.informants, key=lambda p: AssessFitness(p))
            x_plus = best_informant.position
            
            for i in range(dimensions):
                b = random.uniform(0.0, beta)
                c = random.uniform(0.0, gamma)
                d = random.uniform(0.0, delta)
                
                x.velocities[i] = (alpha * x.velocities[i] + 
                                 b * (Best.position[i] - x.position[i]) +
                                 c * (x_plus[i] - x.position[i]) +
                                 d * (x.best_position[i] - x.position[i]))

        for x in P:
            x.position = x.position + epsi * x.velocities

        if AssessFitness(Best) <= target_best:
            print(f"\nSUCCESS! Target fitness reached in {iteration} iterations")
            print(f"Final position: {Best.position}")
            print(f"Final fitness: {AssessFitness(Best)}")
            break

        if iteration % 10 == 0:  # Print progress more frequently
            print(f"\nIteration {iteration}")
            print(f"Current best fitness: {AssessFitness(Best)}")
            print(f"Current best position: {Best.position}")

    return Best

if __name__ == "__main__":
    print("Starting PSO algorithm...")
    best_particle = pso(swarmsize, alpha, beta, gamma, delta, epsi, dimensions)
    print("\nOptimization complete!")
    print("------------------------")
    print(f"Final position: {best_particle.position}")
    print(f"Final fitness: {AssessFitness(best_particle)}")