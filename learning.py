import net
import functions as fun
import numpy as np


def get_delta(layer_input,layer_output,t) :

    delta = []

    for layer in range(net.n_layers):
        delta.append(np.zeros(net.n_layers))

    for layer in range(net.n_layers-1, -1, -1):
        
        act_fun_deriv = fun.activation_functions_deriv[net.act_fun_code_per_layer[layer]]
       
        if layer == net.n_layers-1 :
            error_fun_deriv = fun.error_functions_deriv[net.error_fun_code]

            delta[layer] = act_fun_deriv(layer_input[layer]) * \
                       error_fun_deriv(layer_output[layer],t)
        else :
            delta[layer] = act_fun_deriv(layer_input[layer]) * \
                       np.dot(np.transpose(net.weights[layer+1]),delta[layer+1])
    
    return delta
                       
def get_weights_bias_deriv(x, delta, layer_input, layer_output) :
        
    weights_deriv = []
    bias_deriv = []

    for layer in range(net.n_layers):
        if layer == 0 :
            weights_deriv.append(np.dot(delta[layer],np.transpose(x)))
        else :
            weights_deriv.append(np.dot(delta[layer],np.transpose(layer_output[layer-1])))

        bias_deriv.append(delta[layer]) # right ? 

    return weights_deriv, bias_deriv



def back_progagation(net, x, t):  

    layer_input, layer_output = net.forwardStep(x)

    delta = get_delta(layer_input,layer_output, t)
    weights_deriv, bias_deriv = get_weights_bias_deriv(x, delta, layer_input, layer_output)





