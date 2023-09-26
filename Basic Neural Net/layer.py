from neuron import Neuron

class Layer:
    def __init__(self, neuron_count: int, parent):
        self.parent = parent
        self.neurons = []

    def __iter__(self) -> list[Neuron]:
        return self.neurons
    
    def __call__(self, input) -> list[float]:
        return [neuron(input) for neuron in self] + [-1]


class Input_Layer(Layer):
    def __init__(self, input_parser, size):
        self.input_parser = input_parser
        self.size = size
    
    # This is kind of cursed but like what else could this possibly return
    def __iter__(self):
        return [lambda input: self(input)[i] for i in range(self.size)]
    
    def __call__(self, input) -> list[float]:
        return self.input_parser(input) + [-1]