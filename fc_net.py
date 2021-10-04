import numpy as np
import functions as fun

class MultilayerNet:
    def __init__(self, n_hidden_layers, n_hidden_nodes_per_layer, act_fun_codes, error_fun_code):
        self.n_input_nodes = 784 # dipende dal dataset: mnist_in = 784
        self.n_output_nodes = 10 # dipende dal dataset: mnist_out = 10
        self.n_layers = n_hidden_layers + 1

        self.error_fun_code = error_fun_code
        self.act_fun_code_per_layer = act_fun_codes.copy()

        self.nodes_per_layer = n_hidden_nodes_per_layer.copy()
        self.nodes_per_layer.append(self.n_output_nodes)

        self.weights = list()
        self.bias = list()

        self.__initialize_weights_and_bias()

    def __initialize_weights_and_bias(self):
        mu, sigma = 0, 0.1

        for i in range(self.n_layers):
            if i == 0:
                self.weights.append(np.random.normal(mu, sigma, size=(self.nodes_per_layer[i], self.n_input_nodes)))
            else:
                self.weights.append(np.random.normal(mu, sigma, size=(self.nodes_per_layer[i], self.nodes_per_layer[i-1])))

            self.bias.append(np.random.normal(mu, sigma, size=(self.nodes_per_layer[i], 1)))

    def forward_step(self, x): 
        layers_input = list()
        layers_output = list()

        for i in range(self.n_layers):
            if i == 0:
                input = np.dot(self.weights[i], x) + self.bias[i]
                layers_input.append(input)

            else:
                input = np.dot(self.weights[i], layers_output[i-1]) + self.bias[i]
                layers_input.append(input)

            act_fun = fun.activation_functions[self.act_fun_code_per_layer[i]]
            output = act_fun(input)
            layers_output.append(output)

        return layers_input, layers_output

    def sim(self, x):
        for i in range(self.n_layers):
            if i == 0:
                input = np.dot(self.weights[i], x) + self.bias[i]

            else:
                input = np.dot(self.weights[i], output) + self.bias[i]

            act_fun = fun.activation_functions[self.act_fun_code_per_layer[i]]
            output = act_fun(input)

        return output

    def print_config(self):
        print('\n\n\nYOUR MULTILAYER NETWORK')
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
