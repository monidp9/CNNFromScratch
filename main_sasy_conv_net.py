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

def get_x_t_for_two_classes(X,t): 

    t = np.array(t)
    count = 0 
    for n in range(t.shape[0]):
        label = t[n]
        if label == 1 or label == 2:
            count+=1

    one_hot_labels = np.zeros((2, count))
    count = 0 
    index = list()

    for n in range(t.shape[0]):
        label = t[n]
        if label == 1 or label == 2:
            index.append(n)
            one_hot_labels[label-1][count] = 1
            count+=1
    
    t = one_hot_labels
    X = X[:,index]
    return X,t

def get_x_t_for_two_classes_2(X,t): 

    indexes_1 = np.where(t[0] == 1)
    indexes_2 = np.where(t[1] == 1)

    indexes = np.concatenate((indexes_1[0], indexes_2[0]))
    indexes.sort()

    n_istances = indexes.shape[0]

    t_train = np.zeros((2, n_istances))
    t_train = t[0:2, indexes].copy()

    X_train = np.zeros((784, n_istances))
    X_train = X[:, indexes].copy()

    return 

# caricamento dataset
mndata = MNIST('./python-mnist/data')
X, t = mndata.load_training()

X = utility.get_mnist_data(X)
X = utility.get_scaled_data(X)

# t = utility.get_mnist_labels(t)

X,t = get_x_t_for_two_classes(X,t)

net = ConvolutionalNet(n_cv_layers = 2, n_kernels_per_layer = [1,1], # vincolare il numero max di layer convolutivi
                       n_hidden_nodes = 10, act_fun_codes = [1,2], error_fun_code = 1)

net.print_config()

X,t = utility.get_random_dataset(X,t,1250) 

X_train, X_test, t_train, t_test = utility.train_test_split(X,t,test_size = 0.25)

y_train = net.sim(X_train)

print("errore prima su train: ",fun.error_functions[net.error_fun_code](y_train, t_train))

net = conv_batch_learning(net, X_train, t_train, X_test, t_test)

y_test = net.sim(X_test)

print("errore su test : ",fun.error_functions[net.error_fun_code](y_test, t_test))