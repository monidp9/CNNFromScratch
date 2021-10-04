from functions import cross_entropy
from learning import back_propagation
from learning import batch_learning
import functions as fun 
import utility
import numpy as np
import matplotlib.pyplot as plt

from net import Net
from mnist import MNIST


# caricamento dataset
mndata = MNIST('./python-mnist/data')
X, t = mndata.load_training()

X = utility.get_mnist_data(X)
t = utility.get_mnist_labels(t)

net = Net(n_hidden_layers=2, n_hidden_nodes_per_layer=[10, 10], act_fun_codes=[1, 1, 2], error_fun_code=1)

net.print_config()

X,t = utility.get_random_dataset(X,t,1000) 

X = utility.get_scaled_data(X)

X_train, X_test, t_train, t_test = utility.train_test_split(X,t,0.25)

y_train = net.sim(X_train)
print("errore prima su train: ",fun.error_functions[net.error_fun_code](y_train, t_train))

net = batch_learning(net, X_train, t_train, X_test, t_test)
y_test = net.sim(X_test)

print("errore su test : ",fun.error_functions[net.error_fun_code](y_test, t_test))