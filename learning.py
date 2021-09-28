import net
import functions as fun
import numpy as np
import utility

from copy import deepcopy
from matplotlib.pyplot import prism


DELTA_MAX = 50
DELTA_MIN = 1e-06



def accuracy(y, label):
    pred = list()
    target = list()

    n_instances = y.shape[1]
    for i in range(n_instances):
        pred_value = np.argmax(y[:, i])
        pred.append(pred_value)
        target_value = np.argmax(label[:, i])
        target.append(target_value)

    pred = np.array(pred)
    target = np.array(target)
    return np.sum(pred == target) / len(pred)

# -----------------------------------MULTILAYER NEURAL NETWORK -----------------------------------

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

def __get_weights_bias_deriv(net, x, delta, layer_output) :
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
    weights_deriv, bias_deriv = __get_weights_bias_deriv(net, x, delta, layer_output)

    return weights_deriv, bias_deriv

# -----------------------------------CONVOLUTIONAL NEURAL NETWORK -----------------------------------

class struct_per_RPROP:
    def __init__(self, net):
        self.kernels_deriv = list()
        self.weights_deriv = list()
        self.conv_bias_deriv = list()
        self.full_conn_bias_deriv = list()

        self.kernels_deriv_per_epochs = list()
        self.weights_deriv_per_epochs = list()
        self.conv_bias_deriv_per_epochs = list()
        self.full_conn_bias_deriv_per_epochs = list()

        self.kernels_delta = list()
        self.weights_delta = list()
        self.conv_bias_delta = list()
        self.full_conn_bias_delta = list()

        self.__initialize_delta(net)

    def __initialize_delta(self, net) :

        dim_kernels_per_layer = 1 # primo kernel applicato su input ha dimensione 1
        for i in range(net.n_conv_layers) :
            self.kernels_delta.append(np.random.uniform(size=(net.n_kernels_per_layer[i], dim_kernels_per_layer,
                                                                net.KERNEL_SIZE, net.KERNEL_SIZE)))

            dim_kernels_per_layer = net.n_kernels_per_layer[i]

            n_conv_bias_per_layer = net.get_n_nodes_feature_volume_pre_pooling(i)
            self.conv_bias_delta.append(np.random.uniform(size=(n_conv_bias_per_layer,1)))

        for i in range(net.n_fc_layers):
            if i == 0:
                n_nodes_input = net.get_n_nodes_feature_volume(net.n_conv_layers)
                self.weights_delta.append(np.random.uniform(size=(net.nodes_per_layer[i],
                                                        n_nodes_input)))
            else:
                self.weights_delta.append(np.random.uniform(size=(net.nodes_per_layer[i],
                                                        net.nodes_per_layer[i-1])))

            self.full_conn_bias_delta.append(np.random.uniform(size=(net.nodes_per_layer[i], 1)))

    def set_deriv(self, total_kernels_deriv, total_weights_deriv, total_conv_bias_deriv, total_full_conn_bias_deriv):
        self.kernels_deriv = total_kernels_deriv
        self.weights_deriv = total_weights_deriv
        self.conv_bias_deriv = total_conv_bias_deriv
        self.full_conn_bias_deriv = total_full_conn_bias_deriv

def __convolutional_RPROP(net, struct, eta_n, eta_p, epoch):

    kernels_deriv_prev_epoch = struct.kernels_deriv_per_epochs[epoch-1]
    conv_bias_deriv_prev_epoch = struct.conv_bias_deriv_per_epochs[epoch-1]

    for l in range(net.n_conv_layers) :

        layer_kernels_deriv_prev_epoch = kernels_deriv_prev_epoch[l]
        layer_kernels_deriv = struct.kernels_deriv[l]
        layer_kernels = net.kernels[l]
        layer_kernels_delta = struct.kernels_delta[l]

        for k in range(net.n_kernels_per_layer[l]) :        # h indice che scorre sui vari kernel del layer
            kernels_z_axis_size = layer_kernels_deriv_prev_epoch[k].shape[0]

            for z in range(kernels_z_axis_size) :
                for i in range(net.KERNEL_SIZE) :
                    for j in range(net.KERNEL_SIZE) :

                        if layer_kernels_deriv_prev_epoch[k,z,i,j] * layer_kernels_deriv[k,z,i,j] > 0 :         # problema delta
                            layer_kernels_delta[k,z,i,j] = min(DELTA_MAX, eta_p * layer_kernels_delta[k,z,i,j])

                        if layer_kernels_deriv_prev_epoch[k,z,i,j] * layer_kernels_deriv[k,z,i,j] < 0 :
                            layer_kernels_delta[k,z,i,j] = max(DELTA_MIN, eta_n * layer_kernels_delta[k,z,i,j])

                        layer_kernels[k,z,i,j] = layer_kernels[k,z,i,j] - np.sign(layer_kernels_deriv[k,z,i,j]) * layer_kernels_delta[k,z,i,j]

        layer_conv_bias_deriv_prev_epoch = conv_bias_deriv_prev_epoch[l]
        layer_conv_bias_deriv = struct.conv_bias_deriv[l]
        layer_conv_bias = net.cv_bias[l]
        layer_conv_bias_delta = struct.conv_bias_delta[l]

        for i in range(layer_conv_bias.shape[0]) :
            if layer_conv_bias_deriv_prev_epoch[i] * layer_conv_bias_deriv[i] > 0 :
                layer_conv_bias_delta[i] = min(DELTA_MAX, eta_p * layer_conv_bias_delta[i])

            if layer_conv_bias_deriv_prev_epoch[i] * layer_conv_bias_deriv[i] < 0 :
                layer_conv_bias_delta[i] = max(DELTA_MIN, eta_n * layer_conv_bias_delta[i])

            layer_conv_bias[i] = layer_conv_bias[i] - np.sign(layer_conv_bias_deriv[i]) * layer_conv_bias_delta[i]

    return net

def __full_conn_RPROP(net, struct, eta_n, eta_p, epoch):

    weights_deriv_prev_epoch = struct.weights_deriv_per_epochs[epoch-1]
    full_conn_bias_deriv_prev_epoch = struct.full_conn_bias_deriv_per_epochs[epoch-1]

    for l in range(net.n_fc_layers) :
        layer_weights_deriv_prev_epoch = weights_deriv_prev_epoch[l]
        layer_weights_deriv = struct.weights_deriv[l]
        layer_weights_delta = struct.weights_delta[l]

        layer_full_conn_bias_deriv_prev_epoch = full_conn_bias_deriv_prev_epoch[l]
        layer_full_conn_bias_deriv = struct.full_conn_bias_deriv[l]
        layer_full_conn_bias_delta = struct.full_conn_bias_delta[l]

        layer_weights = net.weights[l]
        layer_full_conn_bias = net.fc_bias[l]

        n_nodes_per_layer = layer_weights_deriv.shape[0]
        n_connections_per_nodes = layer_weights_deriv.shape[1]

        for i in range(n_nodes_per_layer) :
            for j in range(n_connections_per_nodes) :

                if layer_weights_deriv_prev_epoch[i,j] * layer_weights_deriv[i,j] > 0 :
                    layer_weights_delta[i,j] = min(DELTA_MAX, eta_p * layer_weights_delta[i,j])
                if layer_weights_deriv_prev_epoch[i,j] * layer_weights_deriv[i,j] < 0 :
                    layer_weights_delta[i,j] = max(DELTA_MIN,eta_n * layer_weights_delta[i,j])

                layer_weights[i,j] = layer_weights[i,j] - np.sign(layer_weights_deriv[i,j]) * layer_weights_delta[i,j]

            if layer_full_conn_bias_deriv_prev_epoch[i] * layer_full_conn_bias_deriv[i] > 0 :
                layer_full_conn_bias_delta[i] = min(DELTA_MAX, eta_p * layer_full_conn_bias_delta[i])
            if layer_full_conn_bias_deriv_prev_epoch[i] * layer_full_conn_bias_deriv[i] < 0 :
                layer_full_conn_bias_delta[i] = max(DELTA_MIN, eta_n * layer_full_conn_bias_delta[i])

            layer_full_conn_bias[i] = layer_full_conn_bias[i] - np.sign(layer_full_conn_bias_deriv[i]) * layer_full_conn_bias_delta[i]

    return net

def RPROP (net, str_rprop, eta_n, eta_p, epoch):   # serve restituire la rete in ogni funzione(?)

    if epoch==0:
        net = conv_standard_gradient_descent(net, str_rprop, eta_n)
    else :
        net = __convolutional_RPROP(net, str_rprop, eta_n, eta_p, epoch)
        net = __full_conn_RPROP(net, str_rprop, eta_n, eta_p, epoch)

    str_rprop.kernels_deriv_per_epochs.append(str_rprop.kernels_deriv)
    str_rprop.conv_bias_deriv_per_epochs.append(str_rprop.conv_bias_deriv)
    str_rprop.weights_deriv_per_epochs.append(str_rprop.weights_deriv)
    str_rprop.full_conn_bias_deriv_per_epochs.append(str_rprop.full_conn_bias_deriv)

    return net

def conv_batch_learning(net, X_train, t_train, X_val, t_val):
    eta_min = 0.5
    eta_max = 1.2
    n_epochs = 100

    train_errors = list()
    val_errors = list()

    error_fun = fun.error_functions[net.error_fun_code]

    y_val = net.sim(X_val)
    best_net = net
    min_error = error_fun(y_val, t_val)

    total_weights_deriv = None
    total_conv_bias_deriv = None
    total_conv_bias_deriv = None
    total_kernels_deriv = None

    n_instances = X_train.shape[1]

    str_rprop = struct_per_RPROP(net)

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


                for i in range(net.n_fc_layers) :
                    total_weights_deriv[i] = np.add(total_weights_deriv[i], weights_deriv[i])
                    total_full_conn_bias_deriv[i] = np.add(total_full_conn_bias_deriv[i], full_conn_bias_deriv[i])

        str_rprop.set_deriv(total_kernels_deriv,total_weights_deriv,total_conv_bias_deriv,total_full_conn_bias_deriv)

        net = RPROP (net, str_rprop, eta_min, eta_max, epoch)

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

def conv_standard_gradient_descent(net, str_rprop, eta):

    for i in range(net.n_conv_layers):
        net.kernels[i] = net.kernels[i] - (eta * str_rprop.kernels_deriv[i])
        net.cv_bias[i] = net.cv_bias[i] - (eta * str_rprop.conv_bias_deriv[i])

    for i in range(net.n_fc_layers):
        net.weights[i] = net.weights[i] - (eta * str_rprop.weights_deriv[i])
        net.fc_bias[i] = net.fc_bias[i] - (eta * str_rprop.full_conn_bias_deriv[i])

    return net

# ----------------------------------- monty

def __get_fc_delta(net, t, fc_inputs, fc_outputs):
    delta = list()

    n_nodes = net.get_n_nodes_feature_volume(net.n_conv_layers)
    delta.append(np.zeros(n_nodes))
    for i in range(net.n_fc_layers):
        delta.append(np.zeros(net.nodes_per_layer[i]))

    error_fun_deriv = fun.error_functions_deriv[net.error_fun_code]
    for i in range(net.n_fc_layers, -1, -1):
        if i == 0:
            # calcolo delta nodi di input
            delta[i] = 1 * np.dot(np.transpose(net.weights[i]), delta[i+1])

        else:
            act_fun_deriv = fun.activation_functions_deriv[net.act_fun_codes_per_layer[i-1]]
            if i == net.n_fc_layers:
                # calcolo delta nodi di output
                delta[i] = act_fun_deriv(fc_inputs[i-1]) * error_fun_deriv(fc_outputs[i-1], t)

            else:
                # calcolo delta nodi nascosti
                delta[i] = act_fun_deriv(fc_inputs[i-1]) * np.dot(np.transpose(net.weights[i]), delta[i+1])

    return delta

def __get_cv_delta(net, cv_inputs, cv_outputs, flattened_delta):
    # cv_inputs: feature volume a cui è stata applicata la convoluzione (senza ReLU)
    # cv_outputs: feature volume a cui è stato applicato il pooling

    conv_delta = [0] * net.n_conv_layers
    pooling_delta = [0] * net.n_conv_layers

    # riporto i delta dell'ultimo strato nella versione matriciale
    index = 0
    pooling_fv = cv_outputs[net.n_conv_layers - 1]
    depth, rows, columns = pooling_fv.shape[0], \
                           pooling_fv.shape[1], \
                           pooling_fv.shape[2]
    last_layer_delta = np.zeros((depth, rows, columns))
    for d in range(depth):
        for i in range(rows):
            for j in range(columns):
                last_layer_delta[d, i, j] = flattened_delta[index]
                index += 1
    pooling_delta[-1] = last_layer_delta

    # calcolo dei delta
    for l in range(net.n_conv_layers - 1, -1, -1):
        conv_fv = cv_inputs[l]
        pooling_fv = cv_outputs[l]

        last_layer = (l == net.n_conv_layers - 1)

        # calcolo dei delta del layer di pooling (DA RIFARE)
        if not last_layer:
            pooling_delta[l] = np.zeros((pooling_fv.shape[0],
                                         pooling_fv.shape[1],
                                         pooling_fv.shape[2]))

            act_fun = fun.activation_functions[net.CONV_ACT_FUN_CODE]
            succ_conv_delta = conv_delta[l + 1]

            layer_delta = pooling_delta[l]
            # i kernel del layer sono quelli applicati e non quelli da applicare
            layer_kernels = net.kernels[l + 1]

            for d in range(pooling_fv.shape[0]):
                for i in range(pooling_fv.shape[1]):
                    for j in range(pooling_fv.shape[2]):
                        node = pooling_fv[d, i, j]

                        node_delta = act_fun(node)
                        temp_sum = 0
                        for k in range(succ_conv_delta.shape[0]):
                            delta = succ_conv_delta[k, i, j]
                            # prende il peso centrale del kernel map
                            weight = layer_kernels[k, d, 1, 1]
                            temp_sum += delta * weight

                        node_delta = node_delta * temp_sum
                        layer_delta[d, i, j] = node_delta

        # calcolo dei delta del layer di convoluzione
        conv_delta[l] = np.zeros((conv_fv.shape[0],
                                  conv_fv.shape[1],
                                  conv_fv.shape[2]))

        layer_pooling_delta = pooling_delta[l]
        layer_conv_delta = conv_delta[l]
        for d in range(conv_fv.shape[0]):
            for i in range(conv_fv.shape[1]):
                for j in range(conv_fv.shape[2]):
                    node = conv_fv[d, i, j]
                    node_delta = 0

                    index = np.where(node == pooling_fv)
                    is_max = len(index[0]) > 0

                    if is_max:
                        # calcolo delta nel layer di pooling
                        node_delta = layer_pooling_delta[index][0]

                    layer_conv_delta[d, i, j] = node_delta

    return conv_delta

def __get_fc_weights_bias_deriv(net, x, fc_delta, fc_outputs):
    weights_deriv = list()
    bias_deriv = list()

    # nei delta ci sono anche quelli del layer flattened, per questo si esclude
    # la prima posizione con l + 1
    for l in range(net.n_fc_layers):
        if l == 0:
            weights_deriv.append(np.dot(fc_delta[l + 1], np.transpose(x)))
        else:
            weights_deriv.append(np.dot(fc_delta[l + 1], np.transpose(fc_outputs[l-1])))
        bias_deriv.append(fc_delta[l + 1])

    return weights_deriv, bias_deriv

def __get_cv_weights_bias_deriv(net, x, cv_delta, cv_outputs):
    x = utility.convert_to_cnn_input(x, net.MNIST_IMAGE_SIZE)

    kernels_deriv = list()
    bias_deriv = list()

    delta_values = list()
    x_values = list()

    for l in range(net.n_conv_layers):
        conv_delta = cv_delta[l]
        layer_kernels = net.kernels[l]

        prev_pooling_fv = x
        if l != 0:
            prev_pooling_fv = cv_outputs[l - 1]

        padded_pred_pooling_fv = net.padding(prev_pooling_fv)

        # calcolo derivate dei bias
        fv_bias_deriv = conv_delta
        bias_deriv.append(fv_bias_deriv.copy())

        # calcolo derivate dei kernel
        n_kernels = layer_kernels.shape[0]
        kernel = layer_kernels[0, :, :, :]

        k_depth = kernel.shape[0]
        k_rows = kernel.shape[1]
        k_columns = kernel.shape[2]

        layer_kernels_deriv = np.zeros((n_kernels, k_depth, k_rows, k_columns))
        for k in range(n_kernels):
            kernel = layer_kernels[k, :, :, :]

            k_depth = kernel.shape[0]
            k_rows = kernel.shape[1]
            k_columns = kernel.shape[2]

            for d in range(k_depth):
                for r in range(k_rows):
                    for c in range(k_columns):
                        n_rows = padded_pred_pooling_fv.shape[1]
                        n_columns = padded_pred_pooling_fv.shape[2]

                        for i in range(1, n_rows - 1, net.STRIDE):
                            row_start = i - 1
                            row_finish = row_start + k_rows
                            for j in range(1, n_columns - 1, net.STRIDE):
                                column_start = j - 1
                                column_finish = column_start + k_columns

                                region = padded_pred_pooling_fv[d, row_start:row_finish, column_start:column_finish]
                                x = region[r, c]
                                x_values.append(x)

                        n_rows = conv_delta.shape[1]
                        n_columns = conv_delta.shape[2]
                        for i in range(0, n_rows, net.STRIDE):
                            for j in range(0, n_columns, net.STRIDE):
                                delta = conv_delta[k, i, j]
                                delta_values.append(delta)

                        deriv = np.multiply(delta_values, x_values)
                        layer_kernels_deriv[k, d, r, c] = np.sum(deriv)

                        delta_values[:] = []
                        x_values[:] = []

        kernels_deriv.append(layer_kernels_deriv.copy())

    # le derivate sono liste aventi una matrice quadrimensionale per ogni layer
    return kernels_deriv, bias_deriv

def conv_back_propagation(net, x, t):
    # x: singola istanza
    cv_inputs, cv_outputs, fc_inputs, fc_outputs = net.forward_step(x)

    # i delta restituiti sono vettori colonna
    fc_delta = __get_fc_delta(net, t, fc_inputs, fc_outputs)
    # i delta restituiti sono matrici
    conv_delta = __get_cv_delta(net, cv_inputs, cv_outputs, fc_delta[0])

    flattened_layer = cv_outputs[net.n_conv_layers - 1].flatten()
    flattened_layer = flattened_layer.reshape(-1, 1)

    fc_weights_deriv, fc_bias_deriv =  __get_fc_weights_bias_deriv(net, flattened_layer, fc_delta, fc_outputs)
    cv_kernels_deriv, cv_bias_deriv = __get_cv_weights_bias_deriv(net, x, conv_delta, cv_outputs)

    for i in range(net.n_conv_layers):
        cv_bias_deriv[i] = cv_bias_deriv[i].flatten()
        cv_bias_deriv[i] = cv_bias_deriv[i].reshape(-1, 1)

    return fc_weights_deriv, fc_bias_deriv, cv_kernels_deriv, cv_bias_deriv
