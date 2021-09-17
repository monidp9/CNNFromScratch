from functions import identity
import numpy as np
import functions as fun


class Net:
    def __init__(self, n_hidden_layers, n_hidden_nodes_per_layer, act_fun_codes, error_fun_code):
        self.n_input_nodes = 2 # dipende dal dataset
        self.n_output_nodes = 2 # dipende dal dataset
        self.n_layers = n_hidden_layers + 1

        self.error_fun = fun.error_functions[error_fun_code]
        self.error_fun_deriv = fun.error_functions_deriv[error_fun_code]

        self.act_fun_per_layer = list()
        self.act_fun_deriv_per_layer = list()
        for i in act_fun_codes:
            self.act_fun_per_layer.append(fun.activation_functions[i])
            self.act_fun_deriv_per_layer.append(fun.activation_functions_deriv[i])

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

            output = self.act_fun_per_layer[i](layer_input[i])
            layer_output.append(output)

        return layer_input, layer_output


    def print_config(self):
        print('\nYOUR NETORK')
        print('-'*100)

        print(f"• input layer: \t\t {self.n_input_nodes} nodes")

        for i in range(self.n_layers):
            if i != self.n_layers - 1:
                print(f"• hidden layer {i+1}: \t{self.nodes_per_layer[i]} nodes,"
                f"{self.act_fun_per_layer[i]} \t (activation function)")
            else:
                print(f"• output layer: \t{self.n_output_nodes} nodes,"
                f"{self.act_fun_per_layer[i]} \t (activation function)")

        print(f"{self.error_fun} (error function)")

        print('-'*100)
        print('\n')
