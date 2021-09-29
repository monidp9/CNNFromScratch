import numpy as np
from scipy.special import softmax

np.seterr(over='ignore')

# funzioni di attivazione
def identity(x):
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(x, 0)

# derivate funzioni di attivazione
def identity_deriv(x):
    return np.ones(x.shape)

def sigmoid_deriv(x):
    z = sigmoid(x)
    return z * (1 - z)

def relu_deriv(x):
    y = x.copy()
    if isinstance(y, np.ndarray):
        y[y > 0] = 1
        y[y <= 0] = 0
        return y
    return 1 if y > 0 else 0

# funzioni di errore
def sum_of_squares(y, t):
    return 0.5 * np.sum(np.power(y - t, 2))

def cross_entropy(y, t, epsilon=1e-12):
    y = np.clip(y, epsilon, 1. - epsilon)
    return - np.sum(t * np.log(y))

def cross_entropy_softmax(y, t):
    softmax_y = softmax(y, axis=0)
    return cross_entropy(softmax_y, t)

# derivate funzioni di errore
def sum_of_squares_deriv(y, t):
    return y - t

# da verificare
def cross_entropy_deriv(y, t):
    return - t / y

def cross_entropy_softmax_deriv(y, t):
    softmax_y = softmax(y, axis=0)
    return softmax_y - t

def accuracy(y, label):
    pred = list()
    target = list()

    n_instances = y.shape[1]
    for i in range(n_instances):
        pred_value = np.argmax(y[:, i])
        pred.append(pred_value)
        target_value = np.argmax(label[:, i])
        target.append(target_value)

    pred = np.array(pred)
    target = np.array(target)
    return np.sum(pred == target) / len(pred)

activation_functions = [sigmoid, relu, identity]
activation_functions_deriv= [sigmoid_deriv, relu_deriv, identity_deriv]

error_functions = [cross_entropy, cross_entropy_softmax, sum_of_squares]
error_functions_deriv = [cross_entropy_deriv, cross_entropy_softmax_deriv, sum_of_squares_deriv]
