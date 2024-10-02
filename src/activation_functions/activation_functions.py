import numpy as np

# ReLU
def relu(x):
    return np.maximum(x, 0, out=x)

# Hyperbolic Tangent
def tan(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
