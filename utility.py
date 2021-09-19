from functions import identity
import numpy as np
import sys
import os

n_activation_functions = 3
n_error_functions = 2

class NotNumberError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

def types_of_activation_functions():
    print('\n   Types of activation functions:')
    print('   1] sigmoid')
    print('   2] identity')
    print('   3] ReLU\n')

def types_of_error_functions():
    print('\n   Types of error functions:')
    print('   1] Cross Entropy')
    print('   2] Sum of Squares\n')

def check_int_input(value, min_value,max_value):
    if not value.isnumeric():
        raise NotNumberError(value)
    if not int(value) >= min_value or not int(value) <= max_value:
        raise ValueError

def get_int_input(string, min_value, max_value=sys.maxsize) :
    flag = False
    value = None
    while not flag :
        try :
            value = input(string)
            check_int_input(value, min_value, max_value)
            flag = True
        except ValueError:
            print('invalid number!')
        except NotNumberError as e:
            print('invalid value, input must be a positive number!')

    return int(value)

def get_configuration_net():
    print('\n\n\n')
    print('-'*40, 'NEURAL NETWORK PROJECT', '-'*40, '\n'
     '\t\t\t\t creation of a multilayer neural network\n\n\n')


    n_hidden_layers = get_int_input('define the number of hidden layers (min value = 1): ',1)

    print('\nfor each hidden layer define the number of internal nodes and the activation functions.')

    n_hidden_nodes_per_layer = list()
    act_fun_codes = list()

    types_of_activation_functions()

    for i in range(n_hidden_layers):
        print('hidden layer', i+1, ':')

        n_nodes = get_int_input('-  number of nodes: ',1)
        n_hidden_nodes_per_layer.append(int(n_nodes))

        act_fun_code = get_int_input('-  choose activaction function: ',1, n_activation_functions) - 1
        act_fun_codes.append(act_fun_code)

        print('\n')

    print('output layer :')
    act_fun_code = get_int_input('-  choose activaction function: ',1, n_activation_functions) - 1
    act_fun_codes.append(act_fun_code)

    types_of_error_functions()
    error_fun_code = get_int_input('-  define the error function: ',1, n_error_functions) - 1

    os.system('clear')

    return n_hidden_layers, n_hidden_nodes_per_layer, act_fun_codes, error_fun_code

def get_mnist_data(data):
    data = np.array(data)
    data = np.transpose(data)
    return data

def get_mnist_labels(labels):
    labels = np.array(labels)
    one_hot_labels = np.zeros((10, labels.shape[0]))

    for n in range(labels.shape[0]):
        label = labels[n] - 1
        one_hot_labels[label][n] = 1

    return one_hot_labels

def get_iris_labels(labels):
    labels = np.array(labels)
    one_hot_labels = np.zeros((3, labels.shape[0]))

    for n in range(labels.shape[0]):
        label = labels[n] - 1
        one_hot_labels[label][n] = 1

    return one_hot_labels

def get_random_dataset(X, t, n_samples_considered=10000):
    if X.shape[1] < n_samples_considered :
        raise ValueError
        
    n_tot_samples = X.shape[1]
    n_samples_not_considered = n_tot_samples - n_samples_considered

    new_dataset = np.array([1] * n_samples_considered + [0] * n_samples_not_considered )
    np.random.shuffle(new_dataset) 

    index = np.where(new_dataset == 1)
    index = np.reshape(index,-1)

    new_X = X[:,index]
    new_t = t[:,index]

    return new_X, new_t

def train_test_split(X, t, test_size=0.25):

    n_samples = X.shape[1]
    test_size = int(n_samples * test_size)
    train_size = n_samples - test_size
    
    dataset = np.array([1] * train_size + [0] * test_size) 
    np.random.shuffle(dataset)

    train_index = np.where(dataset == 1)
    train_index = np.reshape(train_index,-1)

    X_train = X[:,train_index]
    t_train = t[:,train_index]

    test_index = np.where(dataset == 0)
    test_index = np.reshape(test_index,-1)

    X_test = X[:,test_index]
    t_test = t[:,test_index]

    return X_train, X_test, t_train, t_test
