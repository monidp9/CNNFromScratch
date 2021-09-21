from functions import cross_entropy
from os import get_terminal_size
from learning import back_propagation
from learning import batch_learning

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


net = Net(n_hidden_layers=3,
          n_hidden_nodes_per_layer=[10,10,10],
          act_fun_codes= [0,0,0,1],
          error_fun_code=1)
net.print_config()

X,t = utility.get_random_dataset(X,t,20)

X_train, X_test, t_train, t_test = utility.train_test_split(X,t,0.25)


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

from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

n_samples = X_train.shape[0]

t_train = np.array(t_train)
one_hot_labels = np.zeros((t_train.shape[0], 3))

for n in range(t_train.shape[0]):
    label = t_train[n] - 1
    one_hot_labels[n][label] = 1

t_train = one_hot_labels


t_test = np.array(t_test)
one_hot_labels = np.zeros((t_test.shape[0], 3))

for n in range(t_test.shape[0]):
    label = t_test[n] - 1
    one_hot_labels[n][label] = 1

t_test = one_hot_labels



clf = MLPClassifier(solver='sgd', alpha=1e-5,
                        hidden_layer_sizes=(200, 200), random_state=0, learning_rate_init=0.1, activation='relu', batch_size=n_samples, max_iter=500)


clf.fit(X_train,t_train)

predicted = clf.predict_proba(X_test)

print(log_loss(t_test,predicted))

'''




