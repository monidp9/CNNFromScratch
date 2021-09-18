import net
import functions as fun
import numpy as np


def get_delta(net, t, layer_input, layer_output) :
    delta = []

    for i in range(net.n_layers):
        delta.append(np.zeros(net.n_layers))

    for i in range(net.n_layers-1, -1, -1):
        act_fun_deriv = fun.activation_functions_deriv[net.act_fun_code_per_layer[i]]
       
        if i == net.n_layers-1 :
            error_fun_deriv = fun.error_functions_deriv[net.error_fun_code]

            delta[i] = act_fun_deriv(layer_input[i]) * \
                       error_fun_deriv(layer_output[i],t)
        else :
            delta[i] = act_fun_deriv(layer_input[i]) * \
                       np.dot(np.transpose(net.weights[i+1]),delta[i+1])
    
    return delta
                      
def get_weights_bias_deriv(net, x, delta, layer_input, layer_output) :     
    weights_deriv = []
    bias_deriv = []

    for i in range(net.n_layers):
        if i == 0 :
            weights_deriv.append(np.dot(delta[i],np.transpose(x)))
        else :
            weights_deriv.append(np.dot(delta[i],np.transpose(layer_output[i-1])))

        bias_deriv.append(delta[i])

    return weights_deriv, bias_deriv


def back_progagation(net, x, t):  
    layer_input, layer_output = net.forwardStep(x)
    delta = get_delta(net, t, layer_input,layer_output)
    weights_deriv, bias_deriv = get_weights_bias_deriv(net, x, delta, layer_input, layer_output)





