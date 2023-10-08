from neuron import Neuron
from layer import Layer, Input_Layer

class Neural_Net:
    def __init__(self, input_parser, input_size, output_size, hidden_layer_count, layer_size, cost_function):
        self.layers = [Input_Layer(input_parser, input_size)]
        for i in range(hidden_layer_count):
            self.layers.append(Layer(layer_size, self.layers[-1]))
        self.layers.append(Layer(output_size, self.layers[-1]))
        self.cost_function = cost_function

    def __iter__(self):
        return iter(self.layers)

    def __getitem__(self, index):
        return self.layers[index]
    
    def __setitem__(self, index, value):
        self.layers[index] = value

    def __call__(self, input):
        self.layers[0].update(input)
        for layer in self:
            layer.update(input)
        return self.layers[-1]()[:-1]
    
    def update_gradient(self, input, d):
        def cost():
            return self.cost_function(self(input[0]), input[1])
        for layer in self[1:]:
            layer.update_gradient(cost, d)

    def descend(self, step):
        for layer in self[1:]:
            layer.descend(step)

    def epoch(self, training_data, d=2**(-8), step=2**(-8)):
        for input in training_data:
            self.update_gradient(input, d)
            self.descend(step)