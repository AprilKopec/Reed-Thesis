from neuron import Neuron

class Layer:
    def __init__(self, neuron_count: int, parent):
        self.parent = parent
        self.neurons = [Neuron(self.parent) for _ in range(neuron_count)]

    def __iter__(self) -> list[Neuron]:
        return iter(self.neurons)
    
    def __len__(self):
        return len(self.neurons)

    def update(self, input):
        for neuron in self.neurons:
            assert isinstance(neuron, Neuron)
            neuron.update()

    def __call__(self) -> list[float]:
        return [neuron() for neuron in self] + [-1]
    
    def update_gradient(self, cost, d):
        for neuron in self.neurons:
            assert isinstance(neuron, Neuron)
            neuron.update_gradient(cost, d)

    def descend(self, step):
        for neuron in self:
            neuron.descend(step)


class Input_Layer(Layer):
    def __init__(self, input_parser, size):
        self.input_parser = input_parser
        self.size = size
    
    # This is kind of cursed but like what else could this possibly return
    def __iter__(self):
        return iter([lambda input: self.update(input)[i] for i in range(self.size)])
    
    def __len__(self):
        return self.size
    
    def update(self, input):
        self.activations = self.input_parser(input)
        return self.activations
    
    def __call__(self) -> list[float]:
        return self.activations + [-1]
    
    def update_gradient(self, cost, d):
        assert False