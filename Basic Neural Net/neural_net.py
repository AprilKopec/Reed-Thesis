from neuron import Neuron
from layer import Layer, Input_Layer

class Neural_Net:
    def __init__(self, input_parser, input_parser_output_size, layer_count, layer_size):
        self.layers = [Input_Layer(input_parser)]

        for i in range(layer_count-1):
            self.layers.append(Layer(layer_size, self.layers[-1]))

    def __call__(self, input):
        return self.layers[-1](input)