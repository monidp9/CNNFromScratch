import net
import functions as fun
import numpy as np
from copy import deepcopy


def __get_delta(net, t, layer_input, layer_output) :
    delta = []

    for i in range(net.n_layers):
        delta.append(np.zeros(net.nodes_per_layer[i]))

    for i in range(net.n_layers-1, -1, -1):
        act_fun_deriv = fun.activation_functions_deriv[net.act_fun_code_per_layer[i]]

        if i == net.n_layers-1 :
            # calcolo delta nodi di output
            error_fun_deriv = fun.error_functions_deriv[net.error_fun_code]
            delta[i] = act_fun_deriv(layer_input[i]) *  error_fun_deriv(layer_output[i], t)

        else :
            # calcolo delta nodi interni
            delta[i] = act_fun_deriv(layer_input[i]) *  np.dot(np.transpose(net.weights[i+1]), delta[i+1])

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

def standard_gradient_descent(net, weights_deriv, bias_deriv, eta):
    for i in range(net.n_layers):
        net.weights[i] = net.weights[i] - (eta * weights_deriv[i])
        net.bias[i] = net.bias[i] - (eta * bias_deriv[i])
    return net

def standard_gradient_descent_per_conv_net(net, kernels_deriv, conv_bias_deriv, eta):
    return net

def RPROP (net, deriv, deriv_per_epochs, eta_n, eta_p, delta, epoch):

    delta_max = 50 
    delta_min = 1e-06

    kernels_deriv = deriv[0]
    weights_deriv= deriv[1]
    conv_bias_deriv = deriv[2]
    full_conn_bias_deriv = deriv[3]

    kernels_deriv_per_epochs = deriv_per_epochs[0]
    weights_deriv_per_epochs = deriv_per_epochs[1]
    conv_bias_deriv_per_epochs = deriv_per_epochs[2]
    full_conn_bias_deriv_per_epochs = deriv_per_epochs[3]

    kernels_delta = delta[0]
    weights_delta = delta[1]
    conv_bias_delta = delta[2]
    full_conn_bias_delta = delta[3]

    if epoch==0:
        net = standard_gradient_descent_per_conv_net(net, kernels_deriv, conv_bias_deriv, eta_n)
        kernels_deriv_per_epochs.append(kernels_deriv)
        conv_bias_deriv_per_epochs.append(conv_bias_deriv)

        net = standard_gradient_descent_per_conv_net(net, weights_deriv, full_conn_bias_deriv, eta_p)
        weights_deriv_per_epochs.append(weights_deriv)
        full_conn_bias_deriv_per_epochs.append(full_conn_bias_deriv)

    else :
        kernels_deriv_prev_epoch = kernels_deriv_per_epochs[epoch-1]
        conv_bias_deriv_prev_epoch = conv_bias_deriv_per_epochs[epoch-1]

        weights_deriv_prev_epoch = weights_deriv_per_epochs[epoch-1]
        full_conn_bias_deriv_prev_epoch = full_conn_bias_deriv_per_epochs[epoch-1]

        for l in range(net.n_conv_layers) :
            
            layer_kernels_deriv_prev_epoch = kernels_deriv_prev_epoch[l]
            layer_kernels_deriv = kernels_deriv[l]
            layer_kernels = net.kernels[l]
            layer_kernels_delta = kernels_delta[l]

            for k in range(net.n_kernels_per_layer[l]) :
                for i in range(net.KERNEL_SIZE) :
                    for j in range(net.KERNEL_SIZE) :

                        if layer_kernels_deriv_prev_epoch[k,i,j] * layer_kernels_deriv[k,i,j] > 0 :
                            layer_kernels_delta[k,i,j] = min(delta_max, eta_p * layer_kernels_delta[k,i,j])

                        if layer_kernels_deriv_prev_epoch[k,i,j] * layer_kernels_deriv[k,i,j] < 0 :
                            layer_kernels_delta[k,i,j] = max(delta_min, eta_n * layer_kernels_delta[k,i,j])

                        layer_kernels[k,i,j] = layer_kernels[k,i,j] - np.sign(layer_kernels_deriv[k,i,j]) * layer_kernels_delta[k,i,j]


            layer_conv_bias_deriv_prev_epoch = conv_bias_deriv_prev_epoch[l]
            layer_conv_bias_deriv = conv_bias_deriv[l]
            layer_conv_bias = net.conv_bias[l]
            layer_conv_bias_delta = conv_bias_delta[l]

            for i in range(layer_conv_bias.shape[0]) :
                if layer_conv_bias_deriv_prev_epoch[i] * layer_conv_bias_deriv[i] > 0 :
                    layer_conv_bias_delta[i] = min(delta_max, eta_p * layer_conv_bias_delta[i])

                if layer_kernels_deriv_prev_epoch[i] * layer_conv_bias_deriv[i] < 0 :
                    layer_conv_bias_delta[i] = max(delta_min, eta_n * layer_conv_bias_delta[i])

                layer_conv_bias[i] = layer_conv_bias[i] - np.sign(layer_conv_bias_deriv[i]) * layer_conv_bias_delta[i]

        for l in range(net.n_full_conn_layers) :
            layer_weights_deriv_prev_epoch = weights_deriv_prev_epoch[l]
            layer_weights_deriv = weights_deriv[l]
            layer_weights_delta = weights_delta[l]
                        
            layer_full_conn_bias_deriv_prev_epoch = full_conn_bias_deriv_prev_epoch[l]
            layer_full_conn_bias_deriv = full_conn_bias_deriv[l]
            layer_full_conn_bias_delta = full_conn_bias_delta[l]
            
            layer_weights = net.weights[l]           
            layer_full_conn_bias = net.full_conn_bias[l]
            
            n_nodes_per_layer = layer_weights_deriv.shape[0]
            n_connections_per_nodes = layer_weights_deriv.shape[1]

            for i in range(n_nodes_per_layer) :
                for j in range(n_connections_per_nodes) :

                    if layer_weights_deriv_prev_epoch[i,j] * layer_weights_deriv[i,j] > 0 :
                        layer_weights_delta[i,j] = min(delta_max, eta_p * layer_weights_delta[i,j])
                    if layer_weights_deriv_prev_epoch[i,j] * layer_weights_deriv[i,j] < 0 :
                        layer_weights_delta[i,j] = max(delta_min,eta_n * layer_weights_delta[i,j])

                    layer_weights[i,j] = layer_weights[i,j] - np.sign(layer_weights_deriv[i,j]) * layer_weights_delta[i,j]
                
                if layer_full_conn_bias_deriv_prev_epoch[i] * layer_full_conn_bias_deriv[i] > 0 :
                    layer_full_conn_bias_delta[i] = min(delta_max, eta_p * layer_full_conn_bias_delta[i])
                if layer_full_conn_bias_deriv_prev_epoch[i] * layer_full_conn_bias_deriv[i] < 0 :
                    layer_full_conn_bias_delta[i] = max(delta_min, eta_n * layer_full_conn_bias_delta[i])

                layer_full_conn_bias[i] = layer_full_conn_bias[i] - np.sign(layer_weights_deriv[i]) * layer_full_conn_bias_delta[i]

    return net 

def conv_batch_learning(net, X_train, t_train, X_val, t_val):
    eta_min = 0.001
    eta_max = 3
    n_epochs = 300

    train_errors = list()
    val_errors = list()

    error_fun = fun.error_functions[net.error_fun_code]

    y_val = net.sim(X_val)
    best_net = net
    min_error = error_fun(y_val, t_val)
    
    total_weights_deriv = None
    total_bias_deriv = None
    n_instances = X_train.shape[1]

    # definizione delta 
    delta = list()
    deriv = list()
    deriv_per_epochs = list()

    for i in range(4):
        x = []
        delta.append(x)
        deriv_per_epochs.append(x)

    for i in range(net.n_conv_layers) :
        delta[0].append(np.random.uniform(size=(net.n_kernels_per_layer[i],
                                                net.KERNEL_SIZE, net.KERNEL_SIZE)))                 #kernels delta

        n_conv_bias_per_layer = net.get_n_nodes_feature_volume_pre_pooling(i)
        delta[2].append(np.random.uniform(size=(n_conv_bias_per_layer,1)))                          #conv_bias_delta

    for i in range(net.n_full_conn_layers):

        if i == 0:
            n_nodes_input = net.get_n_nodes_feature_volume(net.n_conv_layers)
            delta[1].append(np.random.uniform(size=(net.nodes_per_layer[i],                         #weights_delta
                                                        n_nodes_input)))
        else:
            delta[1].append(np.random.uniform(size=(net.nodes_per_layer[i],
                                                        net.nodes_per_layer[i-1])))                   

        delta[3].append(np.random.uniform(size=(net.nodes_per_layer[i], 1)))                        #full_conn_bias_delta


    for epoch in range(n_epochs):

        # somma delle derivate
        for n in range(n_instances):
            # si estrapolano singole istanze come vettori colonna
            x = X_train[:, n].reshape(-1, 1)
            t = t_train[:, n].reshape(-1, 1)
            weights_deriv, full_conn_bias_deriv, kernels_deriv, conv_bias_deriv = conv_back_propagation(net, x, t)

            if n == 0:
                total_weights_deriv = deepcopy(weights_deriv)
                total_full_conn_bias_deriv = deepcopy(full_conn_bias_deriv)
                total_conv_bias_deriv = deepcopy(conv_bias_deriv)
                total_kernels_deriv = deepcopy(kernels_deriv)

            else:
                for i in range(net.n_conv_layers):
                    total_kernels_deriv[i] = np.add(total_kernels_deriv[i], kernels_deriv[i])
                    total_conv_bias_deriv[i] = np.add(total_conv_bias_deriv[i], conv_bias_deriv[i])

                for i in range(net.n_full_conn_layers) :
                    total_weights_deriv[i] = np.add(total_weights_deriv[i], weights_deriv[i])
                    total_full_conn_bias_deriv[i] = np.add(total_full_conn_bias_deriv[i], full_conn_bias_deriv[i])


        deriv.append(total_kernels_deriv,total_weights_deriv,total_conv_bias_deriv,total_full_conn_bias_deriv)

        net =  RPROP (net, deriv, deriv_per_epochs, eta_min, eta_max, delta, epoch)

        y_train = net.sim(X_train)
        y_val = net.sim(X_val)
        
        train_error = error_fun(y_train, t_train)
        val_error = error_fun(y_val, t_val)

        train_errors.append(train_error)
        val_errors.append(val_error)

        print('epoch {}, train error {}, val error {}'.format(epoch, train_error, val_error))

        if val_error < min_error:
            min_error = val_error
            best_net = deepcopy(net)
    
    return best_net

def batch_learning(net, X_train, t_train, X_val, t_val):
    eta = 0.001
    n_epochs = 300

    train_errors = list()
    val_errors = list()

    error_fun = fun.error_functions[net.error_fun_code]

    y_val = net.sim(X_val)
    best_net = net
    min_error = error_fun(y_val, t_val)

    total_weights_deriv = None
    total_bias_deriv = None
    n_instances = X_train.shape[1]

    for epoch in range(n_epochs):

        # somma delle derivate
        for n in range(n_instances):
            # si estrapolano singole istanze come vettori colonna
            x = X_train[:, n].reshape(-1, 1)
            t = t_train[:, n].reshape(-1, 1)
            weights_deriv, bias_deriv = back_propagation(net, x, t)

            if n == 0:
                total_weights_deriv = deepcopy(weights_deriv)
                total_bias_deriv = deepcopy(bias_deriv)
            else:
                for i in range(net.n_layers):
                    total_weights_deriv[i] = np.add(total_weights_deriv[i], weights_deriv[i])
                    total_bias_deriv[i] = np.add(total_bias_deriv[i], bias_deriv[i])

        net = standard_gradient_descent(net, total_weights_deriv, total_bias_deriv, eta)

        y_train = net.sim(X_train)
        y_val = net.sim(X_val)
        
        train_error = error_fun(y_train, t_train)
        val_error = error_fun(y_val, t_val)

        train_errors.append(train_error)
        val_errors.append(val_error)

        print('epoch {}, train error {}, val error {}'.format(epoch, train_error, val_error))

        if val_error < min_error:
            min_error = val_error
            best_net = deepcopy(net)

    return best_net

def back_propagation(net, x, t):
    # x: singola istanza
    layer_input, layer_output = net.forward_step(x)
    delta = __get_delta(net, t, layer_input, layer_output)
    weights_deriv, bias_deriv = __get_weights_bias_deriv(net, x, delta, layer_input, layer_output)

    return weights_deriv, bias_deriv

def conv_back_propagation(net,x,t):
    weights_deriv, full_conn_bias_deriv, kernels_deriv, conv_bias_deriv  = 0
    return weights_deriv, full_conn_bias_deriv, kernels_deriv, conv_bias_deriv 