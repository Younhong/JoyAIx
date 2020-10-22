
import numpy as np

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def tanh(x):
    return 2 * sigmoid(2 * x) -1
