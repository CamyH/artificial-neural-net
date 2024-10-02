import numpy as np

# ReLU
def relu(x):
    return np.maximum(x, 0, out=x)
