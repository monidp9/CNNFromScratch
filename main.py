from learning import back_progagation
import utility
import numpy as np
import matplotlib.pyplot as plt

from net import Net
from mnist import MNIST

mndata = MNIST('./python-mnist/data')
images, labels = mndata.load_training()

# utility.get_configuration_net()



image = images[0]
label = labels[0]
image = np.array(image)

def identity(x):
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# creazione rete
net = Net(n_hidden_layers=1,
          n_hidden_nodes=[3],
          activation_functions=[sigmoid, identity],
          error_function=[identity])

back_progagation(net,image,label)
