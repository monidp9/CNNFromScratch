import numpy as np
import functions as fun

# utilizzare una tupla per input nodi-strati

class Net:
    def __init__(self, n_hidden_layers, n_hidden_nodes_per_layer, act_fun_codes, error_fun_code):
        self.n_input_nodes = 4 # dipende dal dataset: 784
        self.n_output_nodes = 3 # dipende dal dataset: 10
        self.n_layers = n_hidden_layers + 1

        self.error_fun_code = error_fun_code
        self.act_fun_code_per_layer = act_fun_codes.copy()

        self.nodes_per_layer = n_hidden_nodes_per_layer.copy()
        self.nodes_per_layer.append(self.n_output_nodes)

        self.weights = list()
        self.bias = list()

        self.__initialize_weights_and_bias()

    def __initialize_weights_and_bias(self):
        for i in range(self.n_layers):
            if i == 0:
                self.weights.append(np.random.uniform(size=(self.nodes_per_layer[i],
                                                     self.n_input_nodes)))
            else:
                self.weights.append(np.random.uniform(size=(self.nodes_per_layer[i],
                                                     self.nodes_per_layer[i-1])))

            self.bias.append(np.random.uniform(size=(self.nodes_per_layer[i], 1)))

    def forward_step(self, x):
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
            output = act_fun(input)
            layer_output.append(output)

        return layer_input, layer_output

    def sim(self, x):
        for i in range(self.n_layers):
            if i == 0:
                # calcolo input dei nodi del primo strato nascosto
                input = np.dot(self.weights[i], x) + self.bias[i]

            else:
                # calcolo input dei nodi di uno strato nascosto generico
                input = np.dot(self.weights[i], output) + self.bias[i]

            act_fun = fun.activation_functions[self.act_fun_code_per_layer[i]]
            output = act_fun(input)

        return output

    def print_config(self):
        print('\nYOUR MULTILAYER NETWORK')
        print('-'*100)

        print("• input layer: {:>11} nodes".format(self.n_input_nodes))

        error_fun = fun.error_functions[self.error_fun_code]
        error_fun = error_fun.__name__

        for i in range(self.n_layers):
            act_fun = fun.activation_functions[self.act_fun_code_per_layer[i]]
            act_fun = act_fun.__name__

            if i != self.n_layers - 1:
                print("• hidden layer {}: {:>8} nodes, ".format(i+1, self.nodes_per_layer[i]),
                "{:^10} \t (activation function)".format(act_fun))

            else:
                print("• output layer: {:>10} nodes, ".format(self.n_output_nodes),
                "{:^10} \t (activation function)".format(act_fun))

        print("\n {} (error function)".format(error_fun))

        print('-'*100)
        print('\n')
