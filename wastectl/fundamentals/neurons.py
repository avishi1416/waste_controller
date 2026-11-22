import numpy as np


def mcculloch_pitts(x, w, threshold):
    """Binary threshold unit: fires (1) when the weighted sum reaches the threshold."""
    return (np.asarray(x) @ np.asarray(w) >= threshold).astype(int)


# classic logic gates as (weights, threshold); NOT uses an inhibitory weight
GATES = {
    "AND": ([1, 1], 2),
    "OR": ([1, 1], 1),
    "NOT": ([-1], 0),
}


def step(z):
    return (z >= 0).astype(float)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def tanh(z):
    return np.tanh(z)


def sigmoid_neuron(x, w, b):
    return sigmoid(np.asarray(x) @ np.asarray(w) + b)
