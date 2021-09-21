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

        self.act_fun_codes = act_fun_codes.copy()
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

            n_nodes_per_conv_layer = self.__get_n_nodes_feature_volume(i)
            self.conv_bias.append(np.random.uniform(size=(n_nodes_per_conv_layer, 1))) 

    def __initialize_weights_and_full_conn_bias(self):
        for i in range(self.n_full_conn_layers):
            if i == 0:
                feature_map_size = self.__get_feature_map_size(self.n_conv_layers)
                n_nodes_input = np.power(feature_map_size,2) * self.n_kernels_per_layers[self.n_conv_layers-1]
                
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
            output_conv_op = round((W - F + 2*P) / S) + 1                      # output operazione convoluzione
            output_max_pooling_op = round((output_conv_op - F) / F) + 1        # output operazione max pooling 
            W = output_max_pooling_op

        n_nodes = np.power(W,2) * self.n_kernels_per_layers[n_conv_layer-1]
        return n_nodes

    def __get_feature_map_size(self, n_conv_layer): 
        W = self.MNIST_IMAGE_SIZE           
        F = self.KERNEL_SIZE                          
        P = self.PADDING                                  
        S = self.STRIDE              

        for i in range(n_conv_layer) :
            output_conv_op = round((W - F + 2*P) / S) + 1                      # output operazione convoluzione
            output_max_pooling_op = round((output_conv_op - F) / F) + 1        # output operazione max pooling 
            W = output_max_pooling_op

        return W

    def __convolution(self, x, kernel, stride=1): # considerare direttamente la costante STRIDE
        kernel = np.array(kernel)  # array in input è sempre array numpy 

        conv_x = list()
        temp_result = list()

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

    def __convolutional_forward_step(self, x):  
        feature_volumes = list()
        conv_inputs = list()                    # non so se serve nella back propagation

        for i in range(self.n_conv_layers) :
            if i == 0 :
                conv_x = self.__convolution(self, x, self.kernels[i])
            else :
                conv_x = self.__convolution(self, feature_volumes[i-1], self.kernels[i])

            conv_inputs.append(conv_x)
            act_fun = fun.activation_functions[self.CONV_ACT_FUN_CODE]
            output = act_fun(conv_x)
            feature_volumes.append(output)

        return conv_inputs, feature_volumes

    def __full_conn_forward_step(self, x) :
        layer_input = list()
        layer_output = list()

        for i in range(self.n_layers):
            if i == 0:
                # calcolo input dei nodi del primo strato nascosto
                input = np.dot(self.weights[i], x) + self.bias[i]
            else:
                # calcolo input dei nodi di uno strato nascosto generico
                input = np.dot(self.weights[i], layer_output[i-1]) + self.bias[i]
           
            layer_input.append(input)
            act_fun = fun.activation_functions[self.act_fun_code_per_layer[i]]
            output = act_fun(input)
            layer_output.append(output)

    def forward_step(self, x):
        x = x.reshape(self.MNIST_IMAGE_SIZE, self.MNIST_IMAGE_SIZE)

        conv_inputs, feature_volumes = self.__convolutional_forward_step(x)

        feature_map_size = self.__get_feature_map_size(self.n_conv_layers-1)                
        input_for_full_conn_forward_step = feature_volumes[self.n_conv_layers-1].reshape(feature_map_size,feature_map_size)

        layer_input, layer_output = self.__full_conn_forward_step(input_for_full_conn_forward_step)

        return conv_inputs, feature_volumes, layer_input, layer_output




        

    def print_config(self):
        pass
