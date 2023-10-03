from neuron import Neuron
from layer import Layer, Input_Layer

class Neural_Net:
    def __init__(self, input_parser, input_parser_output_size, layer_count, layer_size, cost_function):
        self.layers = [Input_Layer(input_parser, input_parser_output_size)]
        for i in range(layer_count-1):
            self.layers.append(Layer(layer_size, self.layers[-1]))
        self.cost_function = cost_function

    def __iter__(self):
        return self.layers

    def __call__(self, input):
        self.layers[0].update(input)
        for layer in self:
            layer.update()
        return self[-1]()[:-1]
    
    def update_gradient(self, input, d):
        def cost():
            return self.cost_function(input, self(input))
        for layer in self:
            layer.update_gradient(cost, d)

    def descend(self, step):
        for layer in self:
            layer.descend(step)

    def epoch(self, training_data, d=2**(-16), step=2**(-8)):
        for input in training_data:
            self.update_gradient(self, input, d)
            self.descend(self, step)