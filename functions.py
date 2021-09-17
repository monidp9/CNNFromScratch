def identity(x):
    return x

def sigmoid(x):
    return 1 

def sum_of_squares(x):
    return x

def cross_entropy(x):
    return 1 

def identity_derivative(x):
    return x

def sigmoid_derivative(x):
    return 1 

def sum_of_squares_derivative(x):
    return x

def cross_entropy_derivative(x):
    return 1 

activation_functions = [sigmoid, identity]
activation_functions_derivative = [sigmoid_derivative,identity_derivative]
error_functions = [sigmoid, identity]