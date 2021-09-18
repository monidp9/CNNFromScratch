from os import get_terminal_size
from learning import back_propagation
import utility
import numpy as np
import matplotlib.pyplot as plt

from net import Net
from mnist import MNIST


# caricamento dataset
mndata = MNIST('./python-mnist/data')
X, t = mndata.load_training()

net = Net(n_hidden_layers=1,
          n_hidden_nodes_per_layer=[4],
          act_fun_codes=[0,1],
          error_fun_code=1)

net.print_config()

X = utility.get_mnist_data(X)
t = utility.get_mnist_labels(t)

X,t = utility.get_random_dataset(X,t,20)

X_train, X_test, t_train, t_test = utility.train_test_split(X,t,0.25)

istance = X_train[:,0].reshape(-1,1)
label =  t_train[:,0].reshape(-1,1)

w,b = back_propagation(net,istance,label)

print(b[0])
print(b[1])