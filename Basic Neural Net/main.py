from neural_net import Neural_Net
from linear_model import Linear_Model
from training_data import data_split, input_parser, cost_function
from matplotlib import pyplot
import numpy as np

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

def learning_rate_plots():
     training, validation = data_split()
     y = {}
     c = 20
     for lr in [4, 1, 2**-4, 2**-6]:
          print(f'LR: {lr}')
          y[lr] = []
          net = Neural_Net(input_parser, 12, 3, 3, 6, cost_function)
          a = [cost_function(net(d[0]), d[1]) for d in validation]
          y[lr].append(sum(a)/len(a))
          for i in range(c):
               a = [cost_function(net(d[0]), d[1]) for d in validation]
               print(f"Epoch {i} validation loss: {sum(a)/len(a)}")
               net.epoch(training, 2**-12, lr)
               y[lr].append(sum(a)/len(a))
          print(y[lr])
     x = np.linspace(0, 100*c, c+1)
     pyplot.plot(x, y[4], label="4")
     pyplot.plot(x, y[1], label="1")
     pyplot.plot(x, y[2**-4], label="1/16")
     pyplot.plot(x, y[2**-6], label="1/64")
     pyplot.title("Validation loss over time")
     pyplot.xlabel("Training samples")
     pyplot.ylabel("Mean squared error on validation")
     pyplot.legend()
     pyplot.show()

learning_rate_plots()