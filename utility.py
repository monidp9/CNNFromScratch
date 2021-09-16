import numpy as np 

class NotNumberError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

def types_of_activation_functions():
    print('\n   Type of activation function:')
    print('   1] sigmoid')
    print('   2] identity')
    print('   3] ReLU\n')

def check_int_input(value, min_value):
    if not value.isnumeric():
        raise NotNumberError(value)  
    if not int(value) >= min_value: 
        raise ValueError

def get_int_input(string, min_value) :
    flag = False
    value = None
    while not flag :
        try :
            value = input(string)
            check_int_input(value, min_value)
            flag = True
        except ValueError: 
            print('invalid number!')
        except NotNumberError as e:
            print('invalid value, "',e.value,'" is not a number!')   

    return int(value)
 
def get_activation_functions(num):
    print('scegliere funzione')


def get_configuration_net():
    print('\n\n\n')
    print('-------------------------------------NEURAL NETWORK PROJECT------------------------------------- \
          \n                            creation of a multilayer neural network\n\n\n')


    n_hidden_layers = get_int_input('define the number of hidden layers (min value = 1): ',1)

    print('\nfor each hidden layer define the number of internal nodes and the activation functions.\n')

    n_nodes_hidden_layers = np.ndarray(int(n_hidden_layers), dtype=int)
    types_of_activation_functions_hidden_layers = np.ndarray(int(n_hidden_layers), dtype=int)

    for i in range(n_hidden_layers): 
        print('hidden layer ',i+1,':')
    
        n_nodes=get_int_input('-  number of nodes: ',1)

        types_of_activation_functions()

        function = get_int_input('-  choose function: ',1)
        n_nodes_hidden_layers[i] = n_nodes
        types_of_activation_functions_hidden_layers[i] = function
        
        print('\n')

    types_of_activation_functions()
    error_function = get_int_input('define the error function: ',1)

    return n_hidden_layers, n_nodes_hidden_layers, types_of_activation_functions_hidden_layers, error_function
