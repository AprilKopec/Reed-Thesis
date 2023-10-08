from neural_net import Neural_Net
from training_data import training_data

data = training_data(100)

def input_parser(x):
    return list(x)

def cost_function(output, true_value):
    return (output[0]-true_value)**2

net = Neural_Net(input_parser, 2, 1, 2, 4, cost_function)

for i in range(100):
    a = [cost_function(net(d[0]), d[1]) for d in data]
    #print(f"Round {i}: {a}")
    print(f"Average Loss: {sum(a)/len(a)}")
    net.epoch(data, 2**-1, 2**-1)

a = [cost_function(net(d[0]), d[1]) for d in data]
#print(f"Final: {a}")
print(f"Average Loss: {sum(a)/len(a)}")

for i in range(10):
    data = training_data(100)
    a = [cost_function(net(d[0]), d[1]) for d in data]
    print(f"Average Loss on new data: {sum(a)/len(a)}")