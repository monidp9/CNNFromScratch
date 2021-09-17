import net
import functions as fun
import numpy as np


def back_progagation(net, x, t):
    
    layer_input, layer_output = net.forwardStep(x)
    
    #derivative_error_function = net.activation_function_derivative[i](parametro)

    delta = []

    for i in range(net.n_layers):
        delta.append(np.zeros(net.n_layers))
    
    for i in range(net.n_layers,0,-1):
        
        if i == net.n_layers :
            delta[i] = net.activation_function_deriv(layer_input[i-1]) * net.error_function_deriv(layer_output[net.n_layers])
        else :
            delta[i] = net.activation_function_deriv(layer_input[i-1]) * np.dot(delta[i+1],net.weights[i])
