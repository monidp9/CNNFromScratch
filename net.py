import numpy as np


class Net:
    def __init__(self, n_hidden_layers, n_hidden_nodes, activation_functions,
                 error_function):
        self.n_input_nodes = 784 # dipende dal dataset
        self.n_output_nodes = 10 # dipende dal dataset
        self.n_layers = n_hidden_layers + 1
        self.error_function = error_function

        self.nodes_per_layer = n_hidden_nodes.copy()
        self.nodes_per_layer.append(self.n_output_nodes)

        self.activation_functions_per_layer = activation_functions.copy()
        self.weights = list()
        self.bias = list()

        self.__initialize_weights_and_bias()


    def __initialize_weights_and_bias(self):
        for i in range(self.n_layers):
            if i == 0:
                self.weights.append(np.random.normal(size=(self.nodes_per_layer[i],
                                                     self.n_input_nodes)))
            else:
                self.weights.append(np.random.normal(size=(self.nodes_per_layer[i],
                                                     self.nodes_per_layer[i-1])))

            self.bias.append(np.random.normal(size=(self.nodes_per_layer[i], 1)))


    def forwardStep(self, x):
        layer_input = list()
        layer_output = list()

        for i in range(self.n_layers):
            if i == 0:
                # calcolo input dei nodi del primo strato nascosto
                input = np.dot(self.weights[i], x) + self.bias[i]
                layer_input.append(input)

            else:
                # calcolo input dei nodi di uno strato nascosto generico
                input = np.dot(self.weights[i], layer_output[i-1]) + self.bias[i]
                layer_input.append(input)

            output = self.activation_functions_per_layer[i](layer_input[i])
            layer_output.append(output)

        return layer_input, layer_output
