import net
import functions as fun
import numpy as np


def __get_delta(net, t, layer_input, layer_output) :
    delta = []

    for i in range(net.n_layers):
        delta.append(np.zeros(net.n_layers))

    for i in range(net.n_layers-1, -1, -1):
        act_fun_deriv = fun.activation_functions_deriv[net.act_fun_code_per_layer[i]]

        if i == net.n_layers-1 :
            error_fun_deriv = fun.error_functions_deriv[net.error_fun_code]

            delta[i] = act_fun_deriv(layer_input[i]) * \
                       error_fun_deriv(layer_output[i], t)
        else :
            delta[i] = act_fun_deriv(layer_input[i]) * \
                       np.dot(np.transpose(net.weights[i+1]), delta[i+1])

    return delta

def __get_weights_bias_deriv(net, x, delta, layer_input, layer_output) :
    weights_deriv = []
    bias_deriv = []

    for i in range(net.n_layers):
        if i == 0 :
            weights_deriv.append(np.dot(delta[i], np.transpose(x)))
        else :
            weights_deriv.append(np.dot(delta[i], np.transpose(layer_output[i-1])))

        bias_deriv.append(delta[i])

    return weights_deriv, bias_deriv

def __sum_of_deriv():
    pass


def back_propagation(net, x, t):
    layer_input, layer_output = net.forwardStep(x)
    delta = __get_delta(net, t, layer_input,layer_output)
    weights_deriv, bias_deriv = __get_weights_bias_deriv(net, x, delta, layer_input, layer_output)

    return weights_deriv, bias_deriv

def standard_gradient_descent(net, weights_deriv, bias_deriv, eta):
    for i in range(net.n_layers):
        layer_weights = net.weights[i]
        layer_weights_deriv = weights_deriv[i]

        layer_weights = layer_weights - eta * layer_weights_deriv

        layer_bias = net.bias[i]
        layer_bias_deriv = bias_deriv[i]

        layer_bias = layer_bias - eta * layer_bias_deriv

        net.weights[i] = layer_weights
        net.bias[i] = layer_bias
    return net

def batch_learning(net, X_train, t_train, X_val, t_val):
    # possibili iperparametri
    eta = 0.1
    n_epochs = 10

    train_errors = list()
    val_errors = list()

    error_fun = fun.error_functions[net.error_fun_code]

    y_val = net.sim(X_val)
    best_net = net
    min_error = error_fun(y_val, t_val)

    for epoch in range(n_epochs):
        weights_deriv, bias_deriv = back_propagation(net, X_train, t_train)

        # somma delle derivate della funzione di errore

        net = standard_gradient_descent(net, weights_deriv, bias_deriv, eta)

        y_train = net.sim(X_train)
        y_val = net.sim(X_val)

        train_error = error_fun(y_train, t_train)
        val_error = error_fun(y_val, t_val)

        train_errors.append(train_error)
        val_errors.append(val_error)

        if val_error < min_error:
            min_error = val_error
            best_net = net

    return best_net
