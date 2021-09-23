from functions import cross_entropy
from learning import back_propagation
from learning import conv_batch_learning

import utility
import numpy as np
import matplotlib.pyplot as plt

from convolutional_net import ConvolutionalNet
from mnist import MNIST


# caricamento dataset
mndata = MNIST('./python-mnist/data')
X, t = mndata.load_training()

X = utility.get_mnist_data(X)
t = utility.get_mnist_labels(t)


n_conv_layers, n_kernels_per_layer, n_hidden_nodes, act_fun_codes, error_fun_code = utility.get_configuration_conv_net()

net = ConvolutionalNet(n_conv_layers = n_conv_layers, n_kernels_per_layer = n_kernels_per_layer,
                       n_hidden_nodes = 3, act_fun_codes = act_fun_codes, error_fun_code = error_fun_code)

net.print_config()

X,t = utility.get_random_dataset(X,t,20)
X_train, X_test, t_train, t_test = utility.train_test_split(X,t,0.25)

# net = conv_batch_learning(net, X_train, t_train, X_test, t_test)