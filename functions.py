import numpy as np

def softmax(scores):
    return np.exp(scores) / np.sum(np.exp(scores))


# funzioni di attivazione
def identity(x):
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(x, 0)


# derivate funzioni di attivazione
def identity_deriv(x):
    return 1

def sigmoid_deriv(x):
    z = sigmoid(x)
    return z * (1 - z)

def relu_deriv(x):
    y = x.copy()
    y[y > 0] = 1
    y[y <= 0] = 0
    return y


# funzioni di errore
def sum_of_squares(pred, label):
    return 0.5 * np.sum(np.power(pred - label, 2))

def cross_entropy(pred, label):
    return - np.sum(label * np.log(pred))

def cross_entropy_softmax(pred, label):
    softmax_pred = softmax(pred)
    return cross_entropy(softmax_pred, label)


# derivate funzioni di errore
def sum_of_squares_deriv(pred, label):
    return pred - label

def cross_entropy_deriv(pred, label):
    return - pred / label

def cross_entropy_softmax_deriv(pred, label):
    softmax_pred = softmax(pred)
    return softmax_pred - label



activation_functions = [sigmoid, identity, relu]
activation_functions_deriv= [sigmoid_deriv, identity_deriv, relu_deriv]

error_functions = [cross_entropy, cross_entropy_softmax, sum_of_squares]
error_functions_deriv = [cross_entropy_deriv, cross_entropy_softmax_deriv, sum_of_squares_deriv]
