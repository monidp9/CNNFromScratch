import numpy as np


class Net:
    def __init__(self, hidden_layers_num, nodes_num, activation_functions,
                 error_function):
        self.input_nodes_num = 3 # dipende dal dataset
        self.output_nodes_num = 3 # dipende dal dataset
        self.layers_num = hidden_layers_num + 1
        self.error_function = error_function

        self.nodes_num_per_layer = list()
        self.activation_functions_per_layer = list()
        self.weights = list()
        self.bias = list()

        self.__set_nodes_number_per_layer(nodes_num)
        self.__set_activation_functions_per_layer(activation_functions)
        self.__initialize_weights_and_bias()


    def __set_nodes_number_per_layer(self, nodes_num):
        if len(nodes_num) != self.layers_num - 1:
            raise Exception("The number of nodes specified doesn't match with "
            "the number of hidden layers")

        for i in range(len(nodes_num)):
            self.nodes_num_per_layer.append(nodes_num[i])
        self.nodes_num_per_layer.append(self.output_nodes_num)


    def __set_activation_functions_per_layer(self, functions):
        if len(functions) != self.layers_num:
            raise Exception("The number of activation functions doesn't match "
                             "with the number of layers")

        for i in range(len(functions)):
            self.activation_functions_per_layer.append(functions[i])


    def __initialize_weights_and_bias(self):
        for i in range(self.layers_num):
            if i == 0:
                self.weights.append(np.random.normal(
                                        size=(self.nodes_num_per_layer[i],
                                              self.input_nodes_num)))
            else:
                self.weights.append(np.random.normal(
                                        size=(self.nodes_num_per_layer[i],
                                              self.nodes_num_per_layer[i-1])))

            self.bias.append(np.random.normal(size=(self.nodes_num_per_layer[i], 1)))


    def forwardStep(self, x):
        layer_input = list()
        layer_output = list()

        for i in range(self.layers_num):
            if i == 0:
                # calcolo input dei nodi del primo strato nascosto
                layer_input.append(np.dot(self.weights[i], np.transpose(x)) \
                                 + self.bias[i])
            else:
                # calcolo input dei nodi di uno strato nascosto generico
                layer_input.append(np.dot(self.weights[i], layer_output[i-1]) \
                                 + self.bias[i])

            layer_output.append(self.activation_functions_per_layer[i](layer_input[i]))

        return layer_output[i]
