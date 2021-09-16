from net import Net
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def identity(x):
    return x

def sum_of_square():
    pass


net = Net(hidden_layers_num=1,
          nodes_num=[2],
          activation_functions=[sigmoid, identity],
          error_function=sum_of_square)


x = np.random.normal(size=(5, 3))
layer_input, layer_output = net.forwardStep(x)

print(layer_input[1])
