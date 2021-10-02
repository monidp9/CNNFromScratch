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

def get_x_t_for_n_classes(X,t): 

    t = np.array(t)
    count = 0 
    for n in range(t.shape[0]):
        label = t[n]
        if label == 1 or label == 2 or label == 3 or label==4 or label==5 :
            count+=1

    one_hot_labels = np.zeros((5, count))
    count = 0 
    index = list()

    for n in range(t.shape[0]):
        label = t[n]
        if label == 1 or label == 2 or label == 3 or label==4 or label==5  :
            index.append(n)
            one_hot_labels[label-1][count] = 1
            count+=1
    
    t = one_hot_labels
    X = X[:,index]
    return X,t

def testing(net, X_train, t_train, X_test, t_test):
    
    DELTA_MIN = [1e-06] #0]
    DELTA_MAX = [50] # 40, 30]
    ETA_MIN = [0.1, 0.0005, 0.01, 0.02, 0.5, 0.0001, 0.005, 0.05] 
    ETA_MAX = [1.2, 1.5, 0.001] #  2]
    EPOCHS = 20

    # prossimo: 0.01
    # 0.0005, 


    for eta_max in ETA_MAX :
        for delta_min in DELTA_MIN :
            for delta_max in DELTA_MAX :
                    for eta_min in ETA_MIN :
                        net = conv_batch_learning(net, X_train, t_train, X_test, t_test, delta_min, delta_max, eta_min, eta_max, EPOCHS)
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
# t = utility.get_mnist_labels(t)

X = utility.get_scaled_data(X)
X,t = get_x_t_for_n_classes(X,t)
X,t = utility.get_random_dataset(X,t,1000) 

X_train, X_test, t_train, t_test = utility.train_test_split(X,t,test_size = 0.25)

testing(net, X_train, t_train, X_test, t_test)