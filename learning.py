import net
import functions as fun
import numpy as np


def back_progagation(net, x, t):    # non sto considerando bias

    layer_input, layer_output = net.forwardStep(x)

    delta = []
    weights_deriv = []
    bias_deriv = []

    for layer in range(net.n_layers):
        delta.append(np.zeros(net.n_layers))

    for layer in range(net.n_layers-1, -1, -1):
        if layer == net.n_layers-1 :
            delta[layer] = net.act_fun_deriv_per_layer[layer](layer_input[layer]) * \
                       net.error_fun_deriv(layer_output[layer],t)
        else :
            delta[layer] = net.act_fun_deriv_per_layer[layer](layer_input[layer]) * \
                       np.dot(np.transpose(net.weights[layer+1]),delta[layer+1])


    for layer in range(net.n_layers):
        if layer == 0 :
            weights_deriv.append(np.dot(delta[layer],np.transpose(x)))
        else :
            weights_deriv.append(np.dot(delta[layer],np.transpose(layer_output[layer-1])))

        bias_deriv.append(delta[layer])
