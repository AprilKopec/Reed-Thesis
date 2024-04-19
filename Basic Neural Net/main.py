from neural_net import Neural_Net
from linear_model import Linear_Model
from training_data import data_split, input_parser, cost_function
from matplotlib import pyplot

def box_plots():
    training, validation = data_split()
    final_costs = []
    lr = 2**-10
    net = Neural_Net(input_parser, 12, 3, 3, 6, cost_function)
    for i in range(120):
        a = [cost_function(net(d[0]), d[1]) for d in validation]
        print(f"Epoch {i} validation loss: {sum(a)/len(a)}")
        net.epoch(training, 2**-10, lr)

    a = [cost_function(net(d[0]), d[1]) for d in validation]
    final_costs += [sum(a)/len(a)]
    print(f"Final for {lr}: {sum(a)/len(a)}")
    print(final_costs)

    linear_model = Linear_Model(training)

    for d in validation:
        if cost_function(net(d[0]), d[1]) > 0.3:
            print(linear_model(d[0]), net(d[0]), d)
    x = [[cost_function(linear_model(d[0]), d[1]) for d in validation],
         [cost_function(net(d[0]), d[1]) for d in validation]]
    
    pyplot.ylabel('Squared Error')
    pyplot.title('Overall')
    pyplot.boxplot(x, labels = ["Linear Model", "Neural Network"])
    pyplot.show()

    x = [[cost_function(linear_model(d[0]), d[1], 0) for d in validation],
         [cost_function(net(d[0]), d[1], 0) for d in validation]]
    pyplot.ylabel('Squared Error')
    pyplot.title('Math Score')
    pyplot.boxplot(x, labels = ["Linear Model", "Neural Network"])
    pyplot.show()

    x = [[cost_function(linear_model(d[0]), d[1], 1) for d in validation],
         [cost_function(net(d[0]), d[1], 1) for d in validation]]
    pyplot.ylabel('Squared Error')
    pyplot.title('Reading Score')
    pyplot.boxplot(x, labels = ["Linear Model", "Neural Network"])
    pyplot.show()

    x = [[cost_function(linear_model(d[0]), d[1], 2) for d in validation],
         [cost_function(net(d[0]), d[1], 2) for d in validation]]
    pyplot.ylabel('Squared Error')
    pyplot.title('Writing Score')
    pyplot.boxplot(x, labels = ["Linear Model", "Neural Network"])
    pyplot.show()

box_plots()