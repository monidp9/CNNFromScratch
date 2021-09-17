import net

def identity(x):
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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

def get_function_derivative(function):
    if function==sigmoid :
        return sigmoid_derivative
    if function==identity :
        return identity_derivative
    if function==cross_entropy :
        return cross_entropy_derivative
    if function==sum_of_squares :
        return sum_of_squares_derivative
    return None
    
derivateFunzioniAttivazione = [identity_derivative, sigmoid_derivative]

funzioniErrore = [sum_of_squares_derivative, cross_entropy_derivative]

def back_progagation(net, x, t):
    
    layer_input, layer_output = net.forwardStep(x)
    
    #derivative_error_function = net.error_function
    print(type(net.error_function))
    #print(derivative_error_function)    