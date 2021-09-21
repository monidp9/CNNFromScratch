from functions import cross_entropy
from os import get_terminal_size
from learning import back_propagation
from learning import batch_learning

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


net = ConvolutionalNet(n_conv_layers = 3, n_kernels_per_layers = [3,3,3], 
                       n_nodes_hidden_layer = 3, act_fun_codes = [0,0], error_fun_code = [0])
  
net.print_config()

X,t = utility.get_random_dataset(X,t,20)
X_train, X_test, t_train, t_test = utility.train_test_split(X,t,0.25)


'''
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X,t = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, t)
X_train = np.transpose(X_train)
t_train = utility.get_iris_labels(y_train)

X_test = np.transpose(X_test)
y_test = utility.get_iris_labels(y_test)

net = batch_learning(net, X_train, t_train, X_test, y_test)
'''










