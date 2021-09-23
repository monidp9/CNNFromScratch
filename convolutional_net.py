from os import PRIO_USER
from math import sqrt
import numpy as np
import functions as fun

class ConvolutionalNet:
    def __init__(self, n_conv_layers, n_kernels_per_layers, n_hidden_nodes, act_fun_codes, error_fun_code):
        self.n_input_nodes = 784        # dipende dal dataset: 784
        self.n_output_nodes = 10        # dipende dal dataset: 10

        self.n_conv_layers = n_conv_layers
        self.n_kernels_per_layers = n_kernels_per_layers.copy()

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
        self.kernels = list()
        self.conv_bias = list()

        self.__initialize_weights_and_full_conn_bias()
        self.__initialize_kernels_and_conv_bias()


    def __initialize_kernels_and_conv_bias(self):
        for i in range(self.n_conv_layers):
            self.kernels.append(np.random.uniform(size=(self.n_kernels_per_layers[i],
                                                        self.KERNEL_SIZE, self.KERNEL_SIZE)))

            n_nodes_conv_layer = self.__get_n_nodes_feature_volume_pre_pooling(i)
            self.conv_bias.append(np.random.uniform(size=(n_nodes_conv_layer, 1)))

    def __initialize_weights_and_full_conn_bias(self):
        for i in range(self.n_full_conn_layers):
            if i == 0:
                n_nodes_input = self.__get_n_nodes_feature_volume(self.n_conv_layers)

                self.weights.append(np.random.uniform(size=(self.nodes_per_layer[i],
                                                      n_nodes_input)))
            else:
                self.weights.append(np.random.uniform(size=(self.nodes_per_layer[i],
                                                     self.nodes_per_layer[i-1])))

            self.full_conn_bias.append(np.random.uniform(size=(self.nodes_per_layer[i], 1)))

    def __get_n_nodes_feature_volume(self, n_conv_layer):
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
        n_nodes = n_nodes * self.n_kernels_per_layers[n_conv_layer-1]

        return n_nodes

    def __get_n_nodes_feature_volume_pre_pooling(self, n_conv_layer):
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
        n_nodes = n_nodes * self.n_kernels_per_layers[n_conv_layer]

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
        feature_volume = self.__padding(feature_volume)

        if feature_volume.ndim < 3:
            feature_volume = np.expand_dims(feature_volume, axis=0)

        depth = feature_volume.shape[0]
        n_rows = feature_volume.shape[1]
        n_columns = feature_volume.shape[2]

        kernel_depth =  kernels.shape[0]
        kernel_rows = kernels.shape[1]
        kernel_columns = kernels.shape[2]

        # convolution
        results = list()
        conv_feature_volume = list()
        feature_map_temp = list()
        row_temp = list()
        bias_index = 0

        for k in range(kernel_depth):
            kernel = kernels[k, :, :]
            for r in range(1, n_rows - 1, self.STRIDE):
                for c in range(1, n_columns - 1, self.STRIDE):
                    row_start = r - 1
                    column_start = c - 1

                    row_finish = row_start + kernel_rows
                    column_finish = column_start + kernel_columns

                    for d in range(depth):
                        region = feature_volume[d, row_start:row_finish, column_start:column_finish]
                        matrix_prod = np.multiply(region, kernel)
                        if d == 0:
                            matrix_sum = matrix_prod
                        else:
                            matrix_sum = matrix_sum + matrix_prod

                    result = np.sum(matrix_sum) + bias[bias_index]
                    row_temp.append(result)
                    bias_index += 1

                feature_map_temp.append(row_temp.copy())
                row_temp[:] = []

            conv_feature_volume.append(feature_map_temp.copy())
            feature_map_temp[:] = []

        conv_feature_volume = np.array(conv_feature_volume)
        return conv_feature_volume

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

    def forward_step(self, x):
        x = x.reshape(self.MNIST_IMAGE_SIZE, self.MNIST_IMAGE_SIZE)

        conv_inputs, feature_volumes = self.__convolutional_forward_step(x)

        flattened_input = feature_volumes[self.n_conv_layers-1].flatten()
        flattened_input = input_for_full_conn.reshape(-1, 1)

        layer_input, layer_output = self.__full_conn_forward_step(input_for_full_conn)

        return conv_inputs, feature_volumes, layer_input, layer_output

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
        for i in range(len(self.n_kernels_per_layers)):
            print('• conv layer {}: {:>11} kernels'.format(i, self.n_kernels_per_layers[i]))

        error_fun = fun.error_functions[self.error_fun_code]
        error_fun = error_fun.__name__

        print("\n {} (error function)".format(error_fun))

        print('-'*100)
        print('\n')
