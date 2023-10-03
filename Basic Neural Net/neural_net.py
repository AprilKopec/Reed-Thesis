from neuron import Neuron
from layer import Layer, Input_Layer

class Neural_Net:
    def __init__(self, input_parser, input_parser_output_size, layer_count, layer_size, cost_function):
        self.layers = [Input_Layer(input_parser, input_parser_output_size)]
        for i in range(layer_count-1):
            self.layers.append(Layer(layer_size, self.layers[-1]))
        self.cost = cost_function

    def __iter__(self):
        return self.layers

    def __call__(self, input):
        self.layers[0].update(input)
        for layer in self:
            layer.update()
        return self[-1]()[:-1]
    
    def cost_gradient(self, v):
        def delta_cost(v, dv):
            return self.cost(v[i]+ dv[i] for i in range(len(v)))
