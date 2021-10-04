import numpy as np
import sys
import os

n_activation_functions = 3
n_error_functions = 3

def __types_of_activation_functions():
    print('\n   Types of activation functions:')
    print('   1] sigmoid')
    print('   2] identity')
    print('   3] ReLU\n')

def __types_of_error_functions():
    print('\n   Types of error functions:')
    print('   1] Cross Entropy')
    print('   2] Cross Entropy Soft Max')
    print('   3] Sum of Squares\n')

def __check_int_input(value, min_value,max_value):
    if not value.isnumeric() or (not int(value) >= min_value or not int(value) <= max_value) :
        raise ValueError

def __get_int_input(string, min_value=0, max_value=sys.maxsize) :
    flag = False
    value = None
    while not flag :
        try :
            value = input(string)
            __check_int_input(value, min_value, max_value)
            flag = True
        except ValueError:
            print('invalid input!\n')

    return int(value)

def is_standard_conf():
    print('\n\n')
    choice =  __get_int_input('Do you want to use the default configuration? (Y=1 / N=0): ',0,1)
    return choice

def get_conf_ml_net():
    print('\n\n\n')
    print('-'*40, 'NEURAL NETWORK PROJECT', '-'*40, '\n'
     '\t\t\t\t creation of a multilayer neural network\n\n\n')


    n_hidden_layers = __get_int_input('define the number of hidden layers (min value = 1): ',min_value=1)

    print('\nfor each hidden layer define the number of internal nodes and the activation functions.')

    n_hidden_nodes_per_layer = list()
    act_fun_codes = list()

    __types_of_activation_functions()

    for i in range(n_hidden_layers):
        print('hidden layer', i+1, ':')

        n_nodes = __get_int_input('-  number of nodes: ',1)
        n_hidden_nodes_per_layer.append(int(n_nodes))

        act_fun_code = __get_int_input('-  choose activaction function: ',1, n_activation_functions) - 1
        act_fun_codes.append(act_fun_code)

        print('\n')

    print('output layer :')
    act_fun_code = __get_int_input('-  choose activaction function: ',1, n_activation_functions) - 1
    act_fun_codes.append(act_fun_code)

    __types_of_error_functions()
    error_fun_code = __get_int_input('-  define the error function: ',1, n_error_functions) - 1

    os.system('clear')

    return n_hidden_layers, n_hidden_nodes_per_layer, act_fun_codes, error_fun_code

def get_conf_cv_net():
    print('\n\n\n')
    print('-'*40, 'NEURAL NETWORK PROJECT', '-'*40, '\n'
     '\t\t\t\t creation of a convolutional neural network\n\n\n')


    n_cv_layers = __get_int_input('define the number of convolutional layers (min value = 1): ',min_value=1)

    print('\nfor each convolutional layer define the number of kernels\n')

    n_kernels_per_layer = list()
    act_fun_codes = list()

    for i in range(n_cv_layers):
        print('convlutional layer', i+1, ':')

        n_kernels= __get_int_input('-  number of kernels: ',1)
        n_kernels_per_layer.append(int(n_kernels))
   
    print("\nfor full connected layers define the numbers of nodes and the activation function")
    __types_of_activation_functions()

    print('hidden layer :')
    act_fun_code = __get_int_input('-  choose activaction function for hidden layer: ',1, n_activation_functions) - 1
    act_fun_codes.append(act_fun_code)

    n_hidden_nodes= __get_int_input('-  number of nodes: ',1)

    print('\noutput layer :')
    act_fun_code = __get_int_input('-  choose activaction function for output layer: ',1, n_activation_functions) - 1
    act_fun_codes.append(act_fun_code)

    __types_of_error_functions()
    error_fun_code = __get_int_input('-  define the error function: ',1, n_error_functions) - 1

    os.system('clear')

    return n_cv_layers, n_kernels_per_layer, n_hidden_nodes, act_fun_codes, error_fun_code

def get_mnist_data(data):
    data = np.array(data)
    data = np.transpose(data)
    return data

def get_mnist_labels(labels):
    labels = np.array(labels)
    one_hot_labels = np.zeros((10, labels.shape[0]), dtype=int)

    for n in range(labels.shape[0]):
        label = labels[n]
        one_hot_labels[label][n] = 1

    return one_hot_labels

def get_random_dataset(X, t, n_samples=10000):
    if X.shape[1] < n_samples :
        raise ValueError
        
    n_tot_samples = X.shape[1]
    n_samples_not_considered = n_tot_samples - n_samples

    new_dataset = np.array([1] * n_samples + [0] * n_samples_not_considered )
    np.random.shuffle(new_dataset) 

    index = np.where(new_dataset == 1)
    index = np.reshape(index,-1)

    new_X = X[:,index]
    new_t = t[:,index]

    return new_X, new_t

def get_scaled_data(X):
    X = X.astype('float32')
    X = X / 255.0
    return X 

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

def convert_to_cnn_input(X, image_size):
    n_instances = X.shape[1]
    new_X = np.empty(shape=(n_instances, image_size, image_size))

    for i in range(n_instances):
        new_X[i] = X[:, i].reshape(image_size, image_size)

    return new_X