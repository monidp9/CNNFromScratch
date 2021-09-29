from math import sqrt
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import functions as fun
import utility



class ConvolutionalNet:
    def __init__(self, n_cv_layers, n_kernels_per_layer, n_hidden_nodes, act_fun_codes, error_fun_code):
        self.n_input_nodes = 784        # dipende dal dataset: 784
        self.n_output_nodes = 10        # dipende dal dataset: 10

        self.n_cv_layers = n_cv_layers
        self.n_kernels_per_layer = n_kernels_per_layer.copy()

        self.CONV_ACT_FUN_CODE = 1
        self.MNIST_IMAGE_SIZE = 28
        self.KERNEL_SIZE = 3
        self.STRIDE = 1
        self.PADDING = 1
        self.n_fc_layers = 2

        self.act_fun_codes_per_layer = act_fun_codes.copy()
        self.nodes_per_layer = list()
        self.nodes_per_layer.append(n_hidden_nodes)
        self.nodes_per_layer.append(self.n_output_nodes)

        self.error_fun_code = error_fun_code

        self.weights = list()
        self.fc_bias = list()
        self.kernels = list()
        self.cv_bias = list()

        self.__initialize_kernels_and_cv_bias()
        self.__initialize_weights_and_fc_bias()

    def __initialize_kernels_and_cv_bias(self):
        # il primo kernel applicato su input ha dimensione 1
        dim_kernels_per_layer = 1 
        mu, sigma = 0, 0.1

        for i in range(self.n_cv_layers):
            self.kernels.append(np.random.normal(mu, sigma, size=(self.n_kernels_per_layer[i], dim_kernels_per_layer,
                                                                  self.KERNEL_SIZE, self.KERNEL_SIZE)))

            dim_kernels_per_layer = self.n_kernels_per_layer[i]
            n_nodes_conv_layer = self.get_n_nodes_feature_volume_pre_pooling(i)

            print(n_nodes_conv_layer)
            self.cv_bias.append(np.random.normal(mu, sigma, size=(n_nodes_conv_layer, 1)))

    def __initialize_weights_and_fc_bias(self):
        mu, sigma = 0, 0.1
        for i in range(self.n_fc_layers):
            if i == 0:
                n_nodes_input = self.get_n_nodes_feature_volume(self.n_cv_layers)

                self.weights.append(np.random.normal(mu, sigma, size=(self.nodes_per_layer[i],
                                                                      n_nodes_input)))
            else:
                self.weights.append(np.random.normal(mu, sigma, size=(self.nodes_per_layer[i],
                                                                self.nodes_per_layer[i-1])))

            self.fc_bias.append(np.random.normal(mu, sigma, size=(self.nodes_per_layer[i], 1)))

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

    def padding(self, feature_volume):
        # operazione di padding adatta solo con l'applicazione
        # di un filtro di dimensione 3x3

        remove_dim = False
        if feature_volume.ndim < 3:
            feature_volume = np.expand_dims(feature_volume, axis=0)
            remove_dim = True

        depth = feature_volume.shape[0]
        rows = feature_volume.shape[1]
        columns = feature_volume.shape[2]

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

    # creare il bias in forma matriciale e non come vettore colonna
    def __convolution(self, feature_volume, layer_kernels, bias):
        # layer_kernels: matrice quadrimensionale del layer

        feature_volume = self.padding(feature_volume)
        if feature_volume.ndim < 3:
            feature_volume = np.expand_dims(feature_volume, axis=0)

        n_rows = feature_volume.shape[1]
        n_columns = feature_volume.shape[2]

        # convolution
        b_index = 0
        n_kernels = layer_kernels.shape[0]
        feature_map = list()
        feature_map_row = list()
        conv_feature_volume = list()

        for k in range(n_kernels):
            kernel = layer_kernels[k]
            k_rows = kernel.shape[1]
            k_columns = kernel.shape[2]

            for i in range(1, n_rows - 1, self.STRIDE):
                row_start = i - 1
                row_finish = row_start + k_rows

                for j in range(1, n_columns - 1, self.STRIDE):
                    column_start = j - 1
                    column_finish = column_start + k_columns

                    region = feature_volume[:, row_start:row_finish, column_start:column_finish]

                    region = np.multiply(region, kernel)
                    node = np.sum(region) + bias[b_index][0]
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
        feature_map = list()
        feature_map_row = list()

        for d in range(depth):
            for i in range(0, n_rows - 1, stride):
                row_start = i
                row_finish = row_start + stride

                for j in range(0, n_columns - 1, stride):
                    column_start = j
                    column_finish = column_start + stride

                    region = feature_volume[d, row_start:row_finish, column_start:column_finish]
                    max_node = np.max(region)

                    feature_map_row.append(max_node)

                feature_map.append(deepcopy(feature_map_row))
                feature_map_row[:] = []

            pooled_feature_volume.append(feature_map.copy())
            feature_map[:] = []

        return np.array(pooled_feature_volume)

    def __cv_forward_step(self, x):
        cv_inputs = list()
        cv_outputs = list()

        for i in range(self.n_cv_layers) :
            if i == 0 :
                conv_x = self.__convolution(x, self.kernels[i], self.cv_bias[i])
            else :
                conv_x = self.__convolution(cv_outputs[i-1], self.kernels[i], self.cv_bias[i])

            cv_inputs.append(conv_x)
            act_fun = fun.activation_functions[self.CONV_ACT_FUN_CODE]
            output = act_fun(conv_x)

            # teoricamente bisognerebbe applicare la funzione di attivazione che in questo
            # caso è la funzione identità quindi non viene considerata
            pooled_x = self.__max_pooling(output, self.KERNEL_SIZE)

            cv_outputs.append(pooled_x)

        return cv_inputs, cv_outputs

    def __fc_forward_step(self, x) :
        fc_layers_inputs = list()
        fc_layers_outputs = list()

        for i in range(self.n_fc_layers):
            if i == 0:
                # calcolo input dei nodi del primo strato nascosto
                layer_input = np.dot(self.weights[i], x) + self.fc_bias[i]
            else:
                # calcolo input dei nodi di uno strato nascosto generico
                layer_input = np.dot(self.weights[i], fc_layers_outputs[i - 1]) + self.fc_bias[i]

            fc_layers_inputs.append(layer_input)

            act_fun = fun.activation_functions[self.act_fun_codes_per_layer[i]]
            output = act_fun(layer_input)
            fc_layers_outputs.append(output)

        return fc_layers_inputs, fc_layers_outputs

    def forward_step(self, x):
        x = utility.convert_to_cnn_input(x, self.MNIST_IMAGE_SIZE)

        cv_inputs, cv_outputs = self.__cv_forward_step(x)

        input_for_fc = cv_outputs[self.n_cv_layers - 1].flatten()
        input_for_fc = input_for_fc.reshape(-1, 1)

        fc_inputs, fc_outputs = self.__fc_forward_step(input_for_fc)

        return cv_inputs, cv_outputs, fc_inputs, fc_outputs

    def sim(self, X):
        pred_values = list()
        X = utility.convert_to_cnn_input(X, self.MNIST_IMAGE_SIZE)

        n_instances = X.shape[0]
        for i in range(n_instances) :

            _, cv_outputs = self.__cv_forward_step(X[i])

            input_for_fc = cv_outputs[self.n_cv_layers - 1].flatten()
            input_for_fc = input_for_fc.reshape(-1, 1)

            _, fc_outputs = self.__fc_forward_step(input_for_fc)

            pred_values.append(fc_outputs[self.n_fc_layers - 1])

        pred_values = np.array(pred_values)
        pred_values = pred_values.squeeze()
        pred_values = pred_values.transpose()
        
        return pred_values

    def print_config(self):
        print('\nYOUR CONVOLUTIONAL NETWORK')
        print('-'*100)

        print("• input layer: {:>12} nodes".format(self.n_input_nodes))

        print("• conv layers {:>13} layers".format(self.n_cv_layers))

        act_fun = fun.activation_functions[self.act_fun_codes_per_layer[0]]
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
