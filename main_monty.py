import utility
import numpy as np
import matplotlib.pyplot as plt
import functions as fun

from copy import deepcopy
from net import Net
from mnist import MNIST
from learning import conv_batch_learning
from convolutional_net import ConvolutionalNet


net = ConvolutionalNet(n_cv_layers=2,
                       n_kernels_per_layer=[2, 2],
                       n_hidden_nodes=5,
                       act_fun_codes=[1, 2],
                       error_fun_code=1)

net.print_config()

mndata = MNIST('./python-mnist/data')
X, t = mndata.load_training()

X = utility.get_mnist_data(X)
t = utility.get_mnist_labels(t)

X_train, t_train = utility.get_random_dataset(X, t, 50)
X_train, X_val, t_train, t_val = utility.train_test_split(X_train, t_train)

X_train = utility.get_scaled_data(X_train)
X_val = utility.get_scaled_data(X_val)

net = conv_batch_learning(net, X_train, t_train, X_val, t_val)