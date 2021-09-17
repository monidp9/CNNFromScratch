import numpy as np

def identity(x):
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sum_of_squares(x):
    return x

def cross_entropy(x):
    return 1

def relu(x):
    return x

def identity_deriv(x):
    return x

def sigmoid_deriv(x):
    return 1

def sum_of_squares_deriv(x):
    return x

def cross_entropy_deriv(x):
    return 1

def relu_deriv(x):
    return 1

activation_functions = [sigmoid, identity, relu]
activation_functions_deriv= [sigmoid_deriv, identity_deriv, relu_deriv]

error_functions = [cross_entropy, sum_of_squares]
error_functions_deriv = [cross_entropy_deriv, sum_of_squares_deriv]
