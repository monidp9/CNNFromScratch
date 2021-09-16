import utility
import numpy as np
import matplotlib.pyplot as plt

from net import Net
from mnist import MNIST

def identity(x):
    return x


mndata = MNIST('./python-mnist/data')
images, labels = mndata.load_training()

# utility.get_configuration_net()

image = images[0]
image = np.array(image)


# creazione rete
net = Net(hidden_layers_num=1,
          nodes_num=[3],
          activation_functions=[identity, identity],
          error_function=[identity])

x = image.reshape(1, -1)

input, output = net.forwardStep(x)

print(output)
