import math
import random as rand

def dot_product(u, v):
    assert len(u) == len(v), "Dot product vectors must be the same length"
    return sum([u[i] * v[i] for i in range(len(u))])

# Activation functions
def ReLU(self, x: float) -> float:
    return max(0, x)

def sigmoid(self, x: float) -> float:
    return 1/(1+math.exp(-x))


class Neuron:
    activation_function = sigmoid

    def __init__(self, weights: list[float], parent_layer: list):
        self.weights = [1-(2*rand.random()) for neuron in parent_layer]
        self.parent_layer = parent_layer
        self.activation = 0

    def update(self):
        self.activation = dot_product(self.parent_layer(), self.weights)

    def __call__(self) -> float:
        return self.activation