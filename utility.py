import numpy as np
import sys
import os

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

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
    choice =  __get_int_input('Do you want to use the default configuration? (Y=1 / N=0): ',0,1)
    os.system('clear')
    return choice

def get_nn_type():
    text = ("\nSelect a neural network type: \n"
            "1] Multilayer Neural Network \n"
            "2] Convolutional Neural Network \n\n? ")
    nn_type = __get_int_input(text, min_value=1, max_value=2)

    if nn_type != 1 and nn_type != 2:
        raise ValueError('Invalid choice')
    
    os.system('clear')
    return 'fc_nn' if nn_type == 1 else 'cv_nn'

def get_conf_ml_net():
    _, columns = os.popen('stty size', 'r').read().split()
    columns = int(columns)

    title = 'NEURAL NETWORK PROJECT'
    sub_title = 'creation of a multilayer neural network'

    title_space = int((columns - (len(title) + 2)) / 2) 
    sub_title_space = int((columns - len(sub_title)) / 2)

    print('\n\n')
    print('-' * title_space, title, '-' * title_space, ' ' * sub_title_space, sub_title, '\n')

    n_hidden_layers = __get_int_input('define the number of hidden layers (min value = 1): ',min_value=1)

    n_hidden_nodes_per_layer = list()
    act_fun_codes = list()

    __types_of_activation_functions()

    for i in range(n_hidden_layers):
        print('hidden layer', i+1)

        n_nodes = __get_int_input('-  number of nodes: ',1)
        n_hidden_nodes_per_layer.append(int(n_nodes))

        act_fun_code = __get_int_input('-  activaction function: ',1, n_activation_functions) - 1
        act_fun_codes.append(act_fun_code)

        print('\n')

    print('output layer :')
    act_fun_code = __get_int_input('-  activaction function: ',1, n_activation_functions) - 1
    act_fun_codes.append(act_fun_code)

    __types_of_error_functions()
    error_fun_code = __get_int_input('-  define the error function: ',1, n_error_functions) - 1

    os.system('clear')

    return n_hidden_layers, n_hidden_nodes_per_layer, act_fun_codes, error_fun_code

def get_conf_cv_net():
    _, columns = os.popen('stty size', 'r').read().split()
    columns = int(columns)

    title = 'NEURAL NETWORK PROJECT'
    sub_title = 'creation of a convolutional neural network'

    title_space = int((columns - (len(title) + 2)) / 2) 
    sub_title_space = int((columns - len(sub_title)) / 2)

    print('\n\n')
    print('-' * title_space, title, '-' * title_space, ' ' * sub_title_space, sub_title, '\n')

    n_cv_layers = __get_int_input('define the number of convolutional layers (min value = 1): ',min_value=1)

    n_kernels_per_layer = list()
    act_fun_codes = list()

    print('\n')
    for i in range(n_cv_layers):
        print('convlutional layer', i+1)

        n_kernels= __get_int_input('-  number of kernels: ',1)
        n_kernels_per_layer.append(int(n_kernels))
   
    __types_of_activation_functions()

    print('hidden layer :')
    act_fun_code = __get_int_input('-  activaction function: ',1, n_activation_functions) - 1
    act_fun_codes.append(act_fun_code)

    n_hidden_nodes= __get_int_input('-  number of nodes: ',1)

    print('\noutput layer :')
    act_fun_code = __get_int_input('-  activaction function: ',1, n_activation_functions) - 1
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

    new_dataset = np.array([1] * n_samples + [0] * n_samples_not_considered)
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

def get_metric_value(y, t, metric):
    pred = np.argmax(y, axis=0)
    target = np.argmax(t, axis=0)

    pred = pred.tolist()
    target = target.tolist()

    if metric == 'accuracy':
        return accuracy_score(pred, target)
    elif metric == 'precision':
        return precision_score(pred, target, average='macro', zero_division=0)
    elif metric == 'recall':
        return recall_score(pred, target, average='macro', zero_division=0)
    elif metric == 'f1':
        return f1_score(pred, target, average='macro', zero_division=0)

    raise ValueError()

def print_result(y_test, t_test):
    accuracy = get_metric_value(y_test, t_test, 'accuracy')
    precision = get_metric_value(y_test, t_test, 'precision')
    recall = get_metric_value(y_test, t_test, 'recall')
    f1 = get_metric_value(y_test, t_test, 'f1')

    print('\n')
    print('-'*63)
    print('Performance on test set\n')
    print('     accuracy: {:.2f} - precision: {:.2f} - recall: {:.2f} - f1: {:.2f}\n\n'.format(accuracy, precision, recall, f1))

