from functions import identity
import numpy as np
import functions as fun


class Net:
    def __init__(self, n_hidden_layers, n_hidden_nodes_per_layer, act_fun_codes, error_fun_code):
        self.n_input_nodes = 2 # dipende dal dataset: 784
        self.n_output_nodes = 2 # dipende dal dataset: 10
        self.n_layers = n_hidden_layers + 1

        self.error_fun_code = error_fun_code
        self.act_fun_code_per_layer = act_fun_codes.copy()

        self.nodes_per_layer = n_hidden_nodes_per_layer.copy()
        self.nodes_per_layer.append(self.n_output_nodes)

        self.weights = list()
        self.bias = list()

        self.__initialize_weights_and_bias()
        self.activation_function_deriv = identity
        self.error_function_deriv = identity


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

            act_fun = fun.activation_functions[self.act_fun_code_per_layer[i]]
            output = act_fun(layer_input[i])
            layer_output.append(output)

        return layer_input, layer_output


    def print_config(self):
        print('\nYOUR NETWORK')
        print('-'*100)

        print(f"• input layer: \t\t {self.n_input_nodes} nodes")

        error_fun = fun.error_functions[self.error_fun_code]
        for i in range(self.n_layers):
            act_fun = fun.activation_functions[self.act_fun_code_per_layer[i]]
            if i != self.n_layers - 1:
                print(f"• hidden layer {i+1}: \t{self.nodes_per_layer[i]} nodes,"
                f"{act_fun} \t (activation function)")
            else:
                print(f"• output layer: \t{self.n_output_nodes} nodes,"
                f"{act_fun} \t (activation function)")

        print(f"{error_fun} (error function)")

        print('-'*100)
        print('\n')
