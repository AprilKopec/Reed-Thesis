def dot_product(u, v):
    assert len(u) == len(v), "Dot product vectors must be the same length"
    return sum([u[i] * v[i] for i in range(len(u))])

class Neuron:
    def __init__(self, weights: list[float], parents: list):
        self.weights = weights
        self.parents = parents

    def activation_function(self, x):
        return max(0,x)

    def __call__(self):
        input = [n() for n in self.parents]
        x = dot_product(self.weights, input)
        return self.activation_function(x)