from numpy.__config__ import show
from functions import cross_entropy
from learning import back_propagation
from learning import conv_batch_learning
import matplotlib.pyplot as plt
import functions as fun 
import utility
import numpy as np
import matplotlib.pyplot as plt

from convolutional_net import ConvolutionalNet
from mnist import MNIST

def testing(net, X_train, t_train, X_test, t_test):
    
    net = conv_batch_learning(net, X_train, t_train, X_test, t_test)

    return

net = ConvolutionalNet(n_cv_layers = 1, 
                       n_kernels_per_layer = [15], # vincolare il numero max di layer convolutivi
                       n_hidden_nodes = 10, 
                       act_fun_codes = [1,2], 
                       error_fun_code = 1)

net.print_config()

# caricamento dataset
mndata = MNIST('./python-mnist/data')
X, t = mndata.load_training()

X = utility.get_mnist_data(X)
t = utility.get_mnist_labels(t)

X = utility.get_scaled_data(X)
X,t = utility.get_random_dataset(X,t,100) 

X_train, X_test, t_train, t_test = utility.train_test_split(X,t,test_size = 0.25)

testing(net, X_train, t_train, X_test, t_test)