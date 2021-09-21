import numpy as np
import functions as fun


class ConvolutionalNet:
    def __init__(self, n_conv_layers, n_kernels_per_layers, n_nodes_hidden_layer, act_fun_codes, error_fun_code):
        self.n_input_nodes = 784        # dipende dal dataset: 784
        self.n_output_nodes = 10        # dipende dal dataset: 10

        self.n_conv_layers = n_conv_layers
        self.n_kernels_per_layers = n_kernels_per_layers.copy()

        self.KERNEL_SIZE = 3            # kernel uguali di dimensione quadrata
        self.STRIDE = 1                 # S: spostamento
        self.PADDING = 1                # P: padding
        self.n_full_conn_layers = 2     # impostazione rete shallow full connected

        self.act_fun_codes = act_fun_codes.copy()
        self.nodes_per_layer = list()
        self.nodes_per_layer.append(n_nodes_hidden_layer)
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
            self.kernels.append(np.random.uniform(size=(self.KERNEL_SIZE, self.KERNEL_SIZE,
                                                        self.n_kernels_per_layers[i])))

            n_nodes_per_conv_layer = self.__get_n_nodes_feature_map(i)
            self.conv_bias.append(np.random.uniform(size=(n_nodes_per_conv_layer, 1)))

    def __initialize_weights_and_full_conn_bias(self):
        for i in range(self.n_full_conn_layers):
            if i == 0:
                input = self.__get_n_nodes_feature_map(self.n_conv_layers)

                self.weights.append(np.random.uniform(size=(self.nodes_per_layer[i],
                                                      input)))
            else:
                self.weights.append(np.random.uniform(size=(self.nodes_per_layer[i],
                                                     self.nodes_per_layer[i-1])))

            self.full_conn_bias.append(np.random.uniform(size=(self.nodes_per_layer[i], 1)))

    def __get_n_nodes_feature_map(self, n_conv_layer):
        W = round(np.sqrt(self.n_input_nodes))
        F = self.KERNEL_SIZE
        P = self.PADDING
        S = self.STRIDE

        for i in range(n_conv_layer) :
            output_conv_op = round((W - F + 2*P) / S) + 1                      # output operazione convoluzione
            output_max_pooling_op = round((output_conv_op - F) / F) + 1        # output operazione max pooling
            W = output_max_pooling_op

        n_node = np.power(W,2) * self.n_kernels_per_layers[n_conv_layer-1]
        return n_node

    def __padding(self, feature_volume):
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

        return np.array(padded_feature_volume)

    def convolution(self, feature_volume, kernels):
        feature_volume = self.__padding(feature_volume)

        depth = feature_volume.shape[0]
        n_rows = feature_volume.shape[1]
        n_columns = feature_volume.shape[2]

        kernel_depth =  kernels.shape[0]
        kernel_rows = kernels.shape[1]
        kernel_columns = kernels.shape[2]

        # convolution
        results = list()
        matrix_sum = None

        conv_feature_volume = list()
        feature_map_temp = list()
        row_temp = list()

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

                    result = np.sum(matrix_sum)
                    row_temp.append(result)

                feature_map_temp.append(row_temp.copy())
                row_temp[:] = []

            conv_feature_volume.append(feature_map_temp.copy())
            feature_map_temp[:] = []

        return np.array(conv_feature_volume)

    # funzione di attivazione
    def max_pooling(self, x, region_size):
        n_rows = x.shape[0]
        n_columns = x.shape[1]

        row_stride = region_size[0]
        column_stride = region_size[1]

        pooled_x = list()
        temp_result = list()

        for i in range(1, n_rows - 1, row_stride):
            for j in range(1, n_columns - 1, column_stride):
                row_start = i - 1
                column_start = j - 1

                row_finish = row_start + row_stride
                column_finish = column_start + column_stride

                region = x[row_start:row_finish, column_start:column_finish]
                max = np.max(region)

                temp_result.append(max)

            pooled_x.append(temp_result.copy())
            temp_result[:] = []

        return np.array(pooled_x)

    def forward_step(self):
        pass

    def print_config(self):
        print('\nYOUR NETWORK')
        print('-'*100)

        print("• input layer: {:>11} nodes".format(self.n_input_nodes))

        print("• conv layers {:>11} layers".format(self.n_conv_layers))

        act_fun = fun.activation_functions[self.act_fun_codes]
        print("• hidden layer: {:>11} nodes".format(self.nodes_per_layer[0]), \
              "{:^10} \t (activation function)".format(act_fun.__name__))

        print("• output layer: {:>11} nodes".format(self.nodes_per_layer[0]))

        error_fun = fun.error_functions[self.error_fun_code]
        error_fun = error_fun.__name__

        print("\n {} (error function)".format(error_fun))

        print('-'*100)
        print('\n')
