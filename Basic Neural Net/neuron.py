import math
import random as rand

def dot_product(u, v):
    assert len(u) == len(v), "Dot product vectors must be the same length"
    return sum([u[i] * v[i] for i in range(len(u))])

# Activation functions
def ReLU(x: float) -> float:
    return max(0, x)

def sigmoid(x: float) -> float:
    return 1/(1+math.exp(-x))


class Neuron:
    activation_function = ReLU

    def __init__(self, parent_layer: list):
        self.weights = [1-(2*rand.random()) for i in range(len(parent_layer)+1)]
        self.parent_layer = parent_layer
        self.activation = 0
        self.gradient = [0]*len(self.weights)

    def update(self):
        self.activation = Neuron.activation_function(dot_product(self.parent_layer(), self.weights))

    def __call__(self) -> float:
        return self.activation

    def update_gradient(self, cost, d=2**(-16)):
        for i in range(len(self.weights)):
            c0 = cost()
            self.weights[i] += d
            c1 = cost()
            self.weights[i] -= 2*d
            c2 = cost()
            self.weights[i] += d
            self.gradient[i] = ((c1-c0)-(c2-c0)) / (2*d)

    def descend(self, step):
        for i in range(len(self.weights)):
            self.weights[i] -= self.gradient[i] * step