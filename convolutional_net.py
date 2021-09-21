import numpy as np
import functions as fun
from numpy.core.shape_base import _block_check_depths_match


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
            # self.full_conn_bias = da fare

    def __initialize_weights_and_full_conn_bias(self):
        for i in range(self.n_full_conn_layers):
            if i == 0:
                input = self.__get_feature_map_size(self.n_conv_layers-1, self.KERNEL_SIZE,
                                                    self.PADDING, self.STRIDE)

                self.weights.append(np.random.uniform(size=(self.nodes_per_layer[i],
                                                      input)))
            else:
                self.weights.append(np.random.uniform(size=(self.nodes_per_layer[i],
                                                     self.nodes_per_layer[i-1])))

            self.full_conn_bias.append(np.random.uniform(size=(self.nodes_per_layer[i], 1)))

    def __get_feature_map_size(n_conv_layer,F,P,S):
        w = 0                                 # w : dimensione input
        return ((w - F + 2*P) / S) + 1

    def convolution(self, x, kernel, stride=1):
        kernel = np.array(kernel)

        conv_x = list()
        temp_result = list()

        # padding  AGGIUSTA IMAGE
        n_columns = x.shape[1]
        vzeros = np.zeros(n_columns)
        x = np.vstack((x, vzeros))
        x = np.vstack((vzeros, x))

        n_rows = x.shape[0]
        hzeros = np.zeros((n_rows, 1))
        x = np.hstack((x, hzeros))
        x = np.hstack((hzeros, x))

        n_rows = x.shape[0]
        n_columns = x.shape[1]

        for i in range(1, n_rows - 1, stride):
            for j in range(1, n_columns - 1, stride):
                row_start = i - 1
                row_finish = row_start + (kernel.shape[0])
                column_start = j - 1
                column_finish = column_start + (kernel.shape[1])

                region = x[row_start:row_finish, column_start:column_finish]

                result = np.multiply(region, kernel)
                result = np.sum(result)

                temp_result.append(result)

            conv_x.append(temp_result.copy())
            temp_result[:] = []

        return np.array(conv_x)

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
        pass
