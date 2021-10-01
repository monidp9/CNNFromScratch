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
                       n_kernels_per_layer=[1, 1],
                       n_hidden_nodes=10,
                       act_fun_codes=[1, 2],
                       error_fun_code=1)

net.print_config()

mndata = MNIST('./python-mnist/data')
X, t = mndata.load_training()

X = utility.get_mnist_data(X)
t = utility.get_mnist_labels(t)


X, t = utility.get_random_dataset(X, t, 1500)


indexes_1 = np.where(t[0] == 1)
indexes_2 = np.where(t[1] == 1)
indexes_3 = np.where(t[2] == 1)

indexes = np.concatenate((indexes_1[0], indexes_2[0], indexes_3[0]))
indexes.sort()

n_istances = indexes.shape[0]

t_train = np.zeros((3, n_istances))
t_train = t[0:3, indexes].copy()

X_train = np.zeros((784, n_istances))
X_train = X[:, indexes].copy()



X_train, X_val, t_train, t_val = utility.train_test_split(X_train, t_train)
X_train, X_test, t_train, t_test = utility.train_test_split(X_train, t_train)


X_train = utility.get_scaled_data(X_train)
X_val = utility.get_scaled_data(X_val)
X_test = utility.get_scaled_data(X_test)

net = conv_batch_learning(net, X_train, t_train, X_val, t_val)

y_pred = net.sim(X_test)
y_pred = fun.softmax(y_pred, axis=0)

for i in range(y_pred.shape[1]):
    print('pred: ', np.argmax(y_pred[:, i]), 'target: ', np.argmax(t_test[:, i]))

