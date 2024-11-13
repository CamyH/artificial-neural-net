import numpy as np

# ReLU
def relu(x):
    return np.maximum(0, x)

# Hyperbolic Tangent
def tan(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

# Logistic Function
def log(x):
    return 1 / (1 + np.exp(-x))
