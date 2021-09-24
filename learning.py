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

def back_propagation(net, x, t):
    # x: singola istanza
    layer_input, layer_output = net.forward_step(x)
    delta = __get_delta(net, t, layer_input, layer_output)
    weights_deriv, bias_deriv = __get_weights_bias_deriv(net, x, delta, layer_input, layer_output)

    return weights_deriv, bias_deriv

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


# ------------------------------------------------------------------------

def __get_fc_delta(net, t, fc_input, fc_output):
    delta = list()
    for i in range(net.n_full_conn_layers + 1):
        delta.append(np.zeros(net.nodes_per_layer[i]))

    error_fun_deriv = fun.error_functions_deriv[net.error_fun_code]
    for i in range(net.n_full_conn_layers, -1, -1):
        if i == 0:
            # calcolo delta nodi di input
            delta[i] = 1 * np.dot(np.transpose(net.weights[i]), delta[i+1])

        else:
            act_fun_deriv = fun.activation_functions_deriv[net.act_fun_codes_per_layer[i-1]]
            if i == net.n_full_conn_layers:
                # calcolo delta nodi di output
                delta[i] = act_fun_deriv(fc_input[i-1]) * error_fun_deriv(fc_output[i-1], t)

            else:
                # calcolo delta nodi nascosti
                delta[i] = act_fun_deriv(fc_input[i-1]) * np.dot(np.transpose(net.weights[i]), delta[i+1])

    return delta

def __get_conv_delta(net, conv_input, conv_output, flattened_delta):
    # conv_input: layer di pooling senza ReLU (pooling da fare)
    # conv_output: layer convolutivo (convoluzione da fare)

    conv_delta = [0] * net.n_conv_layers
    pooling_delta = [0] * net.n_conv_layers

    # riporto i delta dell'ultimo strato nella versione matriciale
    index = 0
    conv_feature_volume = conv_output[net.n_conv_layers - 1]
    depth, rows, columns = conv_feature_volume.shape[0], \
                           conv_feature_volume.shape[1], \
                           conv_feature_volume.shape[2]
    last_layer_delta = np.zeros((depth, rows, columns))
    for d in range(depth):
        for i in range(rows):
            for j in range(columns):
                last_layer_delta[d, i, j] = flattened_delta[index]
                index += 1
    conv_delta[-1] = last_layer_delta

    # calcolo dei delta
    for l in range(net.n_conv_layers - 1, -1, -1):
        conv_feature_volume = conv_output[l]
        pooling_feature_volume = conv_input[l]

        last_layer = (i == net.n_conv_layers - 1)

        # calcolo dei delta del layer di convoluzione (DA RIFARE)
        if not last_layer:
            delta[l] = np.zeros((conv_feature_volume.shape[0],
                                 conv_feature_volume.shape[1],
                                 conv_feature_volume.shape[2]))

            act_fun = fun.activation_functions[self.CONV_ACT_FUN_CODE]

            succ_conv_feature_volume = conv_input[l + 1]
            succ_pooling_delta = pooling_delta[l + 1]

            layer_delta = conv_delta[l]
            layer_kernels = net.kernels[l]
            for d in range(conv_feature_volume.shape[0]):
                for i in range(conv_feature_volume.shape[1]):
                    for j in range(conv_feature_volume.shape[2]):
                        node = conv_feature_volume[d, i, j]

                        node_delta = act_fun(node)
                        temp_sum = 0
                        for k in range(succ_conv_feature_volume.shape[0]):
                            delta = succ_pooling_delta[k, i, j]

                            kernel = layer_kernels[k, :, :]
                            weight = kernel[1, 1]

                            temp_sum += delta * weight

                        node_delta = node_delta * temp_sum
                        layer_delta[d, i, j] = node_delta

        # calcolo dei delta del layer di pooling
        pooling_delta[l] = np.zeros((pooling_feature_volume.shape[0],
                                     pooling_feature_volume.shape[1],
                                     pooling_feature_volume.shape[2]))

        layer_conv_delta = conv_delta[l]
        layer_pooling_delta = pooling_delta[l]
        for d in range(pooling_feature_volume.shape[0]):
            for i in range(pooling_feature_volume.shape[1]):
                for j in range(pooling_feature_volume.shape[2]):
                    node = pooling_feature_volume[d, i, j]
                    node_delta = 0

                    index = np.where(node == conv_feature_volume)
                    is_max = len(index[0]) > 0

                    if is_max:
                        # calcolo delta nel layer di pooling
                        node_delta = layer_conv_delta[index][0]

                    layer_pooling_delta[d, i, j] = node_delta

    return pooling_delta

def __get_fc_weights_bias_deriv(net, x, delta, layer_output):
    weights_deriv = list()
    bias_deriv = list()

    for l in range(net.n_full_conn_layers):
        if l == 0:
            weights_deriv.append(np.dot(delta[l]), np.transpose(x))
        else:
            weights_deriv.append(np.dot(delta[l]), layer_output[l-1])
        bias_deriv.append(delta[l])

    return weights_deriv, bias_deriv

def __get_conv_weights_bias_deriv(net, conv_delta, conv_input, conv_output):
    kernels_deriv = list()
    bias_deriv = list()

    for l in range(net.n_conv_layers):
        if l == 0:
            pass
        else:
            pooling_fv = conv_input[l]ì
            pred_conv_fv = conv_output[l - 1]
            padded_pred_conv_fv = net.__padding(pred_conv_fv)

            layer_kernels = net.kernels[l]
            n_kernels, n_rows, n_columns = layer_kernels.shape[0], \
                                           layer_kernels.shape[1], \
                                           layer_kernels.shape[2]

            kernel_deriv = np.zeros((n_rows, n_columns))
            for k in range(n_kernels):
                for r in range(n_rows):
                    for c in range(n_columns):
                        mask = np.zeros((n_rows, n_columns))
                        mask[r, c] = 1

                        x_values = list()
                        for i in range()







    return weights_deriv, bias_deriv

def back_propagation_conv(net, x, t):
    # x: singola istanza
    conv_input, conv_output, fc_input, fc_output = net.forward_step(x)

    # i delta restituiti sono vettori colonna
    fc_delta = __get_fc_delta(net, t, fc_input, fc_output)
    # i delta restituiti sono matrici
    conv_delta = __get_conv_delta(net, conv_input, conv_output, fc_delta[0])

    flattened_layer = conv_output[net.n_conv_layers - 1].flatten()

    fc_weights_deriv, fc_bias_deriv =  __get_fc_weights_bias_deriv(net, flattened_layer, fc_delta, fc_output)
    conv_kernel_deriv, conv_bias_deriv = __get_conv_weights_bias_deriv(net, conv_delta, conv_input, conv_output)

    return fc_weights_deriv, fc_bias_deriv, conv_kernel_deriv, conv_bias_deriv
