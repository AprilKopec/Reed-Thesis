from neuron import Neuron

class Layer:
    def __init__(self, neuron_count: int, parent):
        self.parent = parent
        self.neurons = []

    def __iter__(self) -> list[Neuron]:
        return self.neurons
    
    def update(self):
        for neuron in self:
            neuron.update()

    def __call__(self) -> list[float]:
        return [neuron() for neuron in self] + [-1]


class Input_Layer(Layer):
    def __init__(self, input_parser, size):
        self.input_parser = input_parser
        self.size = size
    
    # This is kind of cursed but like what else could this possibly return
    def __iter__(self):
        return [lambda input: self.update(input)[i] for i in range(self.size)]
    
    def update(self, input):
        self.activations = self.input_parser(input)
        return self.activations
    
    def __call__(self) -> list[float]:
        return self.activations + [-1]