from neural_net import Neural_Net
from training_data import data, input_parser, validation
from math import sqrt

def cost_function(output, true_value):
    return sqrt((output[0]-true_value[0]/100.0)**2 + (output[1]-true_value[1]/100.0)**2 + (output[2]-true_value[2]/100.0)**2)

def test1():
    final_costs = []
    for learning_rate in [1, 2**-1, 2**-2, 2**-3, 2**-4, 2**-5, 2**-6, 2**-7, 2**-8, 2**-9, 2**-10]:
        net = Neural_Net(input_parser, 17, 3, 1, 20, cost_function)
        print(f'LEARNING RATE: {learning_rate}')
        for i in range(10):
            a = [cost_function(net(d[0]), d[1]) for d in data]
            #print(f"Round {i}: {a}")
            #print(f"Average Loss: {sum(a)/len(a)}")
            a = [cost_function(net(d[0]), d[1]) for d in validation]
            print(f"Average Loss on validation: {sum(a)/len(a)}")
            net.epoch(data, 2**-8, learning_rate)

        a = [cost_function(net(d[0]), d[1]) for d in validation]
        final_costs += [sum(a)/len(a)]
        #print(f"Final: {a}")
        print(f"Final for {learning_rate}: {sum(a)/len(a)}")
    print(final_costs)

def test2():
    final_costs = []
    for layers in [2, 3, 4]:
        net = Neural_Net(input_parser, 17, 3, layers, 20, cost_function)
        for i in range(10):
            a = [cost_function(net(d[0]), d[1]) for d in data]
            #print(f"Round {i}: {a}")
            #print(f"Average Loss: {sum(a)/len(a)}")
            a = [cost_function(net(d[0]), d[1]) for d in validation]
            print(f"Average Loss on validation: {sum(a)/len(a)}")
            net.epoch(data, 2**-8, 2**-4)

        a = [cost_function(net(d[0]), d[1]) for d in validation]
        final_costs += [sum(a)/len(a)]
        #print(f"Final: {a}")
        print(f"Final for {layers}: {sum(a)/len(a)}")
    print(final_costs)

def test3():
    for layers in [2]:
        print(f"{layers} LAYERS:")
        final_costs = []
        for size in [1, 5, 10, 15]:
            net = Neural_Net(input_parser, 17, 3, layers, size, cost_function)
            for i in range(10):
                a = [cost_function(net(d[0]), d[1]) for d in data]
                #print(f"Round {i}: {a}")
                #print(f"Average Loss: {sum(a)/len(a)}")
                a = [cost_function(net(d[0]), d[1]) for d in validation]
                print(f"Average Loss on validation: {sum(a)/len(a)}")
                net.epoch(data, 2**-8, 2**-4)

            a = [cost_function(net(d[0]), d[1]) for d in validation]
            final_costs += [sum(a)/len(a)]
            #print(f"Final: {a}")
            print(f"Final for {size}: {sum(a)/len(a)}")
        print(final_costs)

def test4():
    final_costs = []
    for layers in [2, 3, 4, 5, 6]:
        net = Neural_Net(input_parser, 17, 3, layers, 5, cost_function)
        for i in range(10):
            a = [cost_function(net(d[0]), d[1]) for d in data]
            a = [cost_function(net(d[0]), d[1]) for d in validation]
            print(f"Average Loss on validation: {sum(a)/len(a)}")
            net.epoch(data, 2**-8, 2**-5)

        a = [cost_function(net(d[0]), d[1]) for d in validation]
        final_costs += [sum(a)/len(a)]
        #print(f"Final: {a}")
        print(f"Final for {layers}: {sum(a)/len(a)}")
    print(final_costs)


def test5():
    final_costs = []
    for learning_rate in [1, 2**-1, 2**-2, 2**-3, 2**-4, 2**-5, 2**-6, 2**-7, 2**-8, 2**-9, 2**-10]:
        net = Neural_Net(input_parser, 17, 3, 3, 5, cost_function)
        print(f'LEARNING RATE: {learning_rate}')
        for i in range(10):
            a = [cost_function(net(d[0]), d[1]) for d in data]
            #print(f"Round {i}: {a}")
            #print(f"Average Loss: {sum(a)/len(a)}")
            a = [cost_function(net(d[0]), d[1]) for d in validation]
            print(f"Average Loss on validation: {sum(a)/len(a)}")
            net.epoch(data, 2**-8, learning_rate)

        a = [cost_function(net(d[0]), d[1]) for d in validation]
        final_costs += [sum(a)/len(a)]
        #print(f"Final: {a}")
        print(f"Final for {learning_rate}: {sum(a)/len(a)}")
    print(final_costs)

def test6():
    net = Neural_Net(input_parser, 17, 3, 3, 5, cost_function)
    for i in range(20):
        a = [cost_function(net(d[0]), d[1]) for d in data]
        #print(f"Round {i}: {a}")
        #print(f"Average Loss: {sum(a)/len(a)}")
        a = [cost_function(net(d[0]), d[1]) for d in validation]
        print(f"Average Loss on validation: {sum(a)/len(a)}")
        if i < 10:
            lr = 2**-8
        elif i < 15:
            lr = 2**-10
        else:
            lr = 2**-12
        net.epoch(data, 2**-8, lr)

    a = [cost_function(net(d[0]), d[1]) for d in validation]
    #print(f"Final: {a}")
    print(f"Final: {sum(a)/len(a)}")

test6()