import pandas as pd

from src.helper_functions.helpers import train_test_split

# Data Pre-Processing
data = pd.read_csv('../data/concrete_data.csv')

# Inputs
X = data[['cement', 'blast_furnace_slag', 'fly_ash', 'water', 'superplasticizer', 'coarse_aggregate', 'fine_aggregate', 'age']].copy()
# Output
Y = data[['concrete_compressive_strength']].copy()

# HyperParam Initialisation
# PSO will handle the weights and biases
nodes = 4
layers = 2
iterations = 0

# Train, Test Split
X_train, X_test = train_test_split(X)
Y_train, Y_test = train_test_split(Y)


