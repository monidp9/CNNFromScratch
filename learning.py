import net
import functions as fun
import numpy as np


def back_progagation(net, x, t):    # non sto considerando bias

    layer_input, layer_output = net.forwardStep(x)

    delta = []
    weights_deriv = []

    for i in range(net.n_layers):
        delta.append(np.zeros(net.n_layers))

    for i in range(net.n_layers-1, -1, -1):
        if i == net.n_layers-1 :
            delta[i] = net.act_fun_deriv_per_layer[i](layer_input[i]) * \
                       net.error_fun_deriv(layer_output[i])
        else :
            delta[i] = net.act_fun_deriv_per_layer[i](layer_input[i]) * \
                       np.dot(np.transpose(net.weights[i+1]),delta[i+1])

    for i in range(net.n_layers):
        print(i)
        if i == 0 :
            weights_deriv.append(np.dot(delta[i],np.transpose(x)))
        else :
            weights_deriv.append(np.dot(delta[i],np.transpose(layer_output[i-1])))
