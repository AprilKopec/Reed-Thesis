from neural_net import Neural_Net
from training_data import data, input_parser, validation, cost_function
from matplotlib import pyplot

def box_plots():
    final_costs = []
    lr = 2**-8
    net = Neural_Net(input_parser, 17, 3, 2, 5, cost_function)
    for _ in range(40):
        a = [cost_function(net(d[0]), d[1]) for d in validation]
        print(f"Average Loss on validation: {sum(a)/len(a)}")
        net.epoch(data, 2**-12, lr)

    a = [cost_function(net(d[0]), d[1]) for d in validation]
    final_costs += [sum(a)/len(a)]
    print(f"Final for {lr}: {sum(a)/len(a)}")
    print(final_costs)
    for d in validation:
        if cost_function(net(d[0]), d[1]) > 0.3:
            print(net(d[0]), d)
    x = [[cost_function(net(d[0]), d[1]) for d in validation],
         [cost_function(net(d[0]), d[1], 0) for d in validation],
         [cost_function(net(d[0]), d[1], 1) for d in validation],
         [cost_function(net(d[0]), d[1], 2) for d in validation]
    ]
    pyplot.boxplot(x)
    pyplot.show()

box_plots()