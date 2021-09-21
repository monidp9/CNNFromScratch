import numpy as np
import functions as fun
from numpy.core.shape_base import _block_check_depths_match


class ConvolutionalNet:
    def __init__(self, n_conv_layers, n_kernels_per_layers, n_nodes_hidden_layer, act_fun_codes, error_fun_code):
        self.n_input_nodes = 784 # dipende dal dataset: 784
        self.n_output_nodes = 10 # dipende dal dataset: 10

        self.n_conv_layers = n_conv_layers 
        self.n_kernels_per_layers = n_kernels_per_layers.copy()

        self.KERNEL_SIZE = 3 # si presuppongono kernel uguali di dimensione quadrata
        self.STRIDE = 1         # S: spostamento
        self.PADDING = 1         # P: padding 


        self.act_fun_codes = act_fun_codes.copy()
        self.nodes_per_layer = list()
        self.nodes_per_layer.append(n_nodes_hidden_layer)
        self.nodes_per_layer.append(self.n_output_nodes)        

        self.error_fun_code = error_fun_code

        self.weights = list()
        self.bias = list()
        self.kernels = list()

        self.__initialize_weights_and_bias()
        self.__initialize_kernels()

    def __initialize_kernels(self): 
        for i in range(self.n_conv_layers):
            self.kernels.append(np.random.uniform(size=(self.KERNEL_SIZE, self.KERNEL_SIZE, 
                                                        self.n_kernels_per_layers[i])))


    def __initialize_weights_and_bias(self):
        for i in range(2):
            if i == 0:
                input = 0 # costruire funzione che calcola input 

                #last_conv_layer_size = self.__get_feature_map_size(input,self.KERNEL_SIZE,self.PADDING,self.STRIDE)
                
                self.weights.append(np.random.uniform(size=(self.nodes_per_layer[i],
                                                      input)))
            else:
                self.weights.append(np.random.uniform(size=(self.nodes_per_layer[i],
                                                     self.nodes_per_layer[i-1])))

            self.bias.append(np.random.uniform(size=(self.nodes_per_layer[i], 1)))

 
    def __get_feature_map_size(w,F,P,S): # w : dimensione input
        return ((w - F + 2*P) / S) + 1    

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


