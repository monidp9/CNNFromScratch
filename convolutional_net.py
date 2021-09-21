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


    def convolution(self, x, kernel, stride=1):
        kernel = np.array(kernel)

        conv_x = list()
        temp_result = list()

        # padding
        n_columns = image.shape[1]
        vzeros = np.zeros(n_columns)
        image = np.vstack((image, vzeros))
        image = np.vstack((vzeros, image))

        n_rows = image.shape[0]
        hzeros = np.zeros((n_rows, 1))
        image = np.hstack((image, hzeros))
        image = np.hstack((hzeros, image))

        n_rows = image.shape[0]
        n_columns = image.shape[1]

        for x in range(1, n_rows - 1, stride):
            for y in range(1, n_columns - 1, stride):
                row_start = x - 1
                row_finish = row_start + (kernel.shape[0])
                column_start = y - 1
                column_finish = column_start + (kernel.shape[1])

                region = x[row_start:row_finish, column_start:column_finish]

                result = np.multiply(region, kernel)
                result = np.sum(result)

                temp_result.append(result)

            conv_x.append(temp_result.copy())
            temp_result[:] = []

        return np.array(conv_x)

    def forward_step(self):
        pass

    def max_pooling(self):
        pass

    def print_config(self):
        pass


