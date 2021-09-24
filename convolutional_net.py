from math import sqrt
from copy import deepcopy
import numpy as np
import functions as fun

class ConvolutionalNet:
    def __init__(self, n_conv_layers, n_kernels_per_layer, n_hidden_nodes, act_fun_codes, error_fun_code):
        self.n_input_nodes = 784        # dipende dal dataset: 784
        self.n_output_nodes = 10        # dipende dal dataset: 10

        self.n_conv_layers = n_conv_layers
        self.n_kernels_per_layer = n_kernels_per_layer.copy()

        self.CONV_ACT_FUN_CODE = 1
        self.MNIST_IMAGE_SIZE = 28
        self.KERNEL_SIZE = 3            # kernel uguali di dimensione quadrata
        self.STRIDE = 1                 # S: spostamento
        self.PADDING = 1                # P: padding
        self.n_full_conn_layers = 2     # impostazione rete shallow full connected

        self.act_fun_codes_per_layer = act_fun_codes.copy()
        self.nodes_per_layer = list()
        self.nodes_per_layer.append(n_hidden_nodes)
        self.nodes_per_layer.append(self.n_output_nodes)

        self.error_fun_code = error_fun_code

        self.weights = list()
        self.full_conn_bias = list()
        self.kernels = list()           # lista di kernels quadrimensionali
        self.conv_bias = list()

        self.__initialize_weights_and_full_conn_bias()
        self.__initialize_kernels_and_conv_bias()

    def __initialize_kernels_and_conv_bias(self):

        dim_kernels_per_layer = 1 # primo kernel applicato su input ha dimensione 1

        for i in range(self.n_conv_layers):
            self.kernels.append(np.random.uniform(size=(self.n_kernels_per_layer[i], dim_kernels_per_layer,
                                                        self.KERNEL_SIZE, self.KERNEL_SIZE)))

            dim_kernels_per_layer = self.n_kernels_per_layer[i]
            n_nodes_conv_layer = self.get_n_nodes_feature_volume_pre_pooling(i)
            self.conv_bias.append(np.random.uniform(size=(n_nodes_conv_layer, 1)))

    def __initialize_weights_and_full_conn_bias(self):
        for i in range(self.n_full_conn_layers):
            if i == 0:
                n_nodes_input = self.get_n_nodes_feature_volume(self.n_conv_layers)

                self.weights.append(np.random.uniform(size=(self.nodes_per_layer[i],
                                                      n_nodes_input)))
            else:
                self.weights.append(np.random.uniform(size=(self.nodes_per_layer[i],
                                                     self.nodes_per_layer[i-1])))

            self.full_conn_bias.append(np.random.uniform(size=(self.nodes_per_layer[i], 1)))

    def get_n_nodes_feature_volume(self, n_conv_layer):
        W = self.MNIST_IMAGE_SIZE
        F = self.KERNEL_SIZE
        P = self.PADDING
        S = self.STRIDE

        for i in range(n_conv_layer) :
            # output operazione convoluzione
            output_conv_op = round((W - F + 2*P) / S) + 1
            # output operazione max pooling
            output_max_pooling_op = round((output_conv_op - F) / F) + 1
            W = output_max_pooling_op

        n_nodes = np.power(W,2)
        n_nodes = n_nodes * self.n_kernels_per_layer[n_conv_layer-1]

        return n_nodes

    def get_n_nodes_feature_volume_pre_pooling(self, n_conv_layer):
        W = self.MNIST_IMAGE_SIZE
        F = self.KERNEL_SIZE
        P = self.PADDING
        S = self.STRIDE

        for i in range(n_conv_layer + 1) :
            if i != n_conv_layer:
                # output operazione convoluzione
                output_conv_op = round((W - F + 2*P) / S) + 1
                # output operazione max pooling
                output_max_pooling_op = round((output_conv_op - F) / F) + 1
                W = output_max_pooling_op
            else:
                output_conv_op = round((W - F + 2*P) / S) + 1
                W = output_conv_op

        n_nodes = np.power(W, 2)
        n_nodes = n_nodes * self.n_kernels_per_layer[n_conv_layer]

        return n_nodes

    def __padding(self, feature_volume):
        remove_dim = False
        if feature_volume.ndim < 3:
            feature_volume = np.expand_dims(feature_volume, axis=0)
            remove_dim = True

        depth = feature_volume.shape[0]
        rows = feature_volume.shape[1]
        columns = feature_volume.shape[2]

        feature_map = None
        padded_feature_volume = list()
        for d in range(depth):
            feature_map = feature_volume[d, :, :]

            n_columns = feature_map.shape[1]
            vzeros = np.zeros(n_columns)

            feature_map = np.vstack((feature_map, vzeros))
            feature_map = np.vstack((vzeros, feature_map))

            n_rows = feature_map.shape[0]
            hzeros = np.zeros((n_rows, 1))

            feature_map = np.hstack((feature_map, hzeros))
            feature_map = np.hstack((hzeros, feature_map))

            padded_feature_volume.append(feature_map)

        padded_feature_volume = np.array(padded_feature_volume)
        if remove_dim:
            padded_feature_volume = np.squeeze(padded_feature_volume, axis=0)

        return padded_feature_volume

    def __convolution(self, feature_volume, kernels, bias):
        # kernels quadrimensionale del layer
        feature_volume = self.__padding(feature_volume)

        if feature_volume.ndim < 3:
            feature_volume = np.expand_dims(feature_volume, axis=0)

        n_rows = feature_volume.shape[1]
        n_columns = feature_volume.shape[2]

        # convolution
        b_index = 0
        n_kernels = kernels.shape[0]
        feature_map = list()
        feature_map_row = list()
        conv_feature_volume = list()

        for k in range(n_kernels):
            kernel = kernels[k]
            k_rows = kernels.shape[1]
            k_columns = kernels.shape[2]

            for i in range(1, n_rows - 1, self.STRIDE):
                row_start = i - 1
                row_finish = row_start + k_rows

                for j in range(1, n_columns - 1, self.STRIDE):
                    column_start = j - 1
                    column_finish = column_start + k_columns

                    region = feature_volume[:, row_start:row_finish, column_start:column_finish]

                    region = np.multiply(region, kernel)
                    node = np.sum(region) + bias[b_index]
                    b_index += 1

                    feature_map_row.append(node)

                feature_map.append(deepcopy(feature_map_row))
                feature_map_row[:] = []

            conv_feature_volume.append(deepcopy(feature_map))
            feature_map[:] = []

        return np.array(conv_feature_volume)

    def __max_pooling(self, feature_volume, region_size):
        depth = feature_volume.shape[0]
        n_rows = feature_volume.shape[1]
        n_columns = feature_volume.shape[2]

        stride = region_size

        pooled_feature_volume = list()
        feature_temp = list()
        row_temp = list()

        for d in range(depth):
            for i in range(1, n_rows - 1, stride):
                for j in range(1, n_columns - 1, stride):
                    row_start = i - 1
                    column_start = j - 1

                    row_finish = row_start + stride
                    column_finish = column_start + stride

                    region = feature_volume[d, row_start:row_finish, column_start:column_finish]
                    max = np.max(region)

                    row_temp.append(max)

                feature_temp.append(row_temp.copy())
                row_temp[:] = []

            pooled_feature_volume.append(feature_temp.copy())
            feature_temp[:] = []

        return np.array(pooled_feature_volume)

    def __convolutional_forward_step(self, x):
        feature_volumes = list()
        conv_inputs = list()

        for i in range(self.n_conv_layers) :
            if i == 0 :
                conv_x = self.__convolution(x, self.kernels[i], self.conv_bias[i])
            else :
                conv_x = self.__convolution(feature_volumes[i-1], self.kernels[i], self.conv_bias[i])

            conv_inputs.append(conv_x)
            act_fun = fun.activation_functions[self.CONV_ACT_FUN_CODE]
            output = act_fun(conv_x)

            # teoricamente bisognerebbe applicare la funzione di attivazione che in questo
            # caso è la funzione identità quindi non viene considerata
            pooled_x = self.__max_pooling(output, self.KERNEL_SIZE)

            feature_volumes.append(pooled_x)

        return conv_inputs, feature_volumes

    def __full_conn_forward_step(self, x) :
        layer_input = list()
        layer_output = list()

        for i in range(self.n_full_conn_layers):
            if i == 0:
                # calcolo input dei nodi del primo strato nascosto
                input = np.dot(self.weights[i], x) + self.full_conn_bias[i]
            else:
                # calcolo input dei nodi di uno strato nascosto generico
                input = np.dot(self.weights[i], layer_output[i-1]) + self.full_conn_bias[i]

            layer_input.append(input)
            act_fun = fun.activation_functions[self.act_fun_codes_per_layer[i]]
            output = act_fun(input)
            layer_output.append(output)

        return layer_input, layer_output

    def forward_step(self, X):
        new_X = np.empty(shape=(X.shape[1],self.MNIST_IMAGE_SIZE,self.MNIST_IMAGE_SIZE))

        tot_conv_inputs = list()
        tot_feature_volumes = list()
        tot_layer_input = list()
        tot_layer_output = list()

        for i in range(X.shape[1]):
            new_X[i,:,:] = X[:,i].reshape(self.MNIST_IMAGE_SIZE, self.MNIST_IMAGE_SIZE)

        for i in range(new_X.shape[0]) :

            conv_inputs, feature_volumes = self.__convolutional_forward_step(new_X[i])         #conv_inputs probabilmente non serve

            input_for_full_conn = feature_volumes[self.n_conv_layers-1].flatten()
            input_for_full_conn = input_for_full_conn.reshape(-1, 1)

            layer_input, layer_output = self.__full_conn_forward_step(input_for_full_conn)

            tot_conv_inputs.append(conv_inputs)
            tot_feature_volumes.append(feature_volumes)
            tot_layer_input.append(layer_input)
            tot_layer_output.append(layer_output)

        return tot_conv_inputs, tot_feature_volumes, tot_layer_input, tot_layer_output

    def sim(self, X):
        new_X = np.empty(shape=(X.shape[1],self.MNIST_IMAGE_SIZE,self.MNIST_IMAGE_SIZE))
        tot_layer_output = list()

        for i in range(X.shape[1]):
            new_X[i,:,:] = X[:,i].reshape(self.MNIST_IMAGE_SIZE, self.MNIST_IMAGE_SIZE)

        for i in range(new_X.shape[0]) :

            _, feature_volumes = self.__convolutional_forward_step(new_X[i])         #conv_inputs probabilmente non serve

            input_for_full_conn = feature_volumes[self.n_conv_layers-1].flatten()
            input_for_full_conn = input_for_full_conn.reshape(-1, 1)

            _, layer_output = self.__full_conn_forward_step(input_for_full_conn)

            tot_layer_output.append(layer_output[self.n_full_conn_layers-1])

        return tot_layer_output

    def print_config(self):
        print('\nYOUR CONVOLUTIONAL NETWORK')
        print('-'*100)

        print("• input layer: {:>12} nodes".format(self.n_input_nodes))

        print("• conv layers {:>13} layers".format(self.n_conv_layers))

        act_fun = fun.activation_functions[self.act_fun_codes_per_layer[0]] # da modificare
        print("• hidden layer: {:>11} nodes".format(self.nodes_per_layer[0]), \
              "{:^10} \t (activation function)".format(act_fun.__name__))

        act_fun = fun.activation_functions[self.act_fun_codes_per_layer[1]]
        print("• output layer: {:>11} nodes".format(self.nodes_per_layer[1]), \
              "{:^10} \t (activation function)".format(act_fun.__name__))

        print('\n')
        for i in range(len(self.n_kernels_per_layer)):
            print('• conv layer {}: {:>11} kernels'.format(i, self.n_kernels_per_layer[i]))

        error_fun = fun.error_functions[self.error_fun_code]
        error_fun = error_fun.__name__

        print("\n {} (error function)".format(error_fun))

        print('-'*100)
        print('\n')
