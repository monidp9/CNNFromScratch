import functions as fun
import numpy as np
from copy import deepcopy

# -----------------------------------MULTILAYER NEURAL NETWORK -----------------------------------

def __get_delta(net, t, layers_input, layers_output) :
    delta = list()
    for i in range(net.n_layers):
        delta.append(np.zeros(net.nodes_per_layer[i]))

    for i in range(net.n_layers-1, -1, -1):
        act_fun_deriv = fun.activation_functions_deriv[net.act_fun_code_per_layer[i]]

        if i == net.n_layers-1 :
            # calcolo delta nodi di output
            error_fun_deriv = fun.error_functions_deriv[net.error_fun_code]
            delta[i] = act_fun_deriv(layers_input[i]) *  error_fun_deriv(layers_output[i], t)

        else :
            # calcolo delta nodi interni
            delta[i] = act_fun_deriv(layers_input[i]) *  np.dot(np.transpose(net.weights[i+1]), delta[i+1])

    return delta

def __get_weights_bias_deriv(net, x, delta, layers_output) :
    weights_deriv = []
    bias_deriv = []

    for i in range(net.n_layers):
        if i == 0 :
            weights_deriv.append(np.dot(delta[i], np.transpose(x)))
        else :
            weights_deriv.append(np.dot(delta[i], np.transpose(layers_output[i-1])))
        bias_deriv.append(delta[i])

    return weights_deriv, bias_deriv

def __standard_gradient_descent(net, weights_deriv, bias_deriv, eta):
    for i in range(net.n_layers):
        net.weights[i] = net.weights[i] - (eta * weights_deriv[i])
        net.bias[i] = net.bias[i] - (eta * bias_deriv[i])
    return net

def batch_learning(net, X_train, t_train, X_val, t_val, eta = 0.001, n_epochs = 500):
    
    train_errors, val_errors = list(), list()

    error_fun = fun.error_functions[net.error_fun_code]

    best_net, min_error = None, None
    tot_weights_deriv, tot_bias_deriv = None, None

    n_instances = X_train.shape[1]

    for epoch in range(n_epochs):
        # somma delle derivate
        for n in range(n_instances):
            # si estrapolano singole istanze come vettori colonna
            x = X_train[:, n].reshape(-1, 1)
            t = t_train[:, n].reshape(-1, 1)
            weights_deriv, bias_deriv = __back_propagation(net, x, t)

            if n == 0:
                tot_weights_deriv = deepcopy(weights_deriv)
                tot_bias_deriv = deepcopy(bias_deriv)
            else:
                for i in range(net.n_layers):
                    tot_weights_deriv[i] = np.add(tot_weights_deriv[i], weights_deriv[i])
                    tot_bias_deriv[i] = np.add(tot_bias_deriv[i], bias_deriv[i])

        net = __standard_gradient_descent(net, tot_weights_deriv, tot_bias_deriv, eta)

        y_train = net.sim(X_train)
        y_val = net.sim(X_val)

        train_error = error_fun(y_train, t_train)
        val_error = error_fun(y_val, t_val)

        train_errors.append(train_error)
        val_errors.append(val_error)

        if epoch % 10 == 0:
            print('epoch {}: train error {}, val error {}, acc {}'.format(epoch, train_error, val_error, fun.accuracy(y_val, t_val)))

        if best_net is None or val_error < min_error:
            min_error = val_error
            best_net = deepcopy(net)

    return best_net

def __back_propagation(net, x, t):
    # x: singola istanza
    layers_input, layers_output = net.forward_step(x)
    delta = __get_delta(net, t, layers_input, layers_output)
    weights_deriv, bias_deriv = __get_weights_bias_deriv(net, x, delta, layers_output)

    return weights_deriv, bias_deriv
