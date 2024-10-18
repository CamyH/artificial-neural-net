from src.activation_functions.activation_functions import relu
from src.pso.pso import particle_swarm_optimisation
import pandas as pd

if __name__ == '__main__':
    data = pd.read_csv('./data/concrete_data.csv')

    particle_swarm_optimisation(data.shape[0], data['concrete_compressive_strength'].values , 5, 7)