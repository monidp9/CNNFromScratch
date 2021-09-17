from functions import identity
import numpy as np 
import sys 

n_activation_functions = 3
n_error_functions = 2

class NotNumberError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

def types_of_activation_functions():
    print('\n   Types of activation functions:')
    print('   1] sigmoid')
    print('   2] identity')
    print('   3] ReLU\n')

def types_of_error_functions(): 
    print('\n   Types of error functions:')
    print('   1] Cross Entropy')
    print('   2] Sum of Squares\n')
    
def get_activation_function(num):

    if num==1 :
        return 'sigmoid'
    if num==2 :
        return 'identity'
    if num==3 :
        return 'ReLU'

def get_error_function(num):

    if num==1 :
        return 'cross_entropy'
    if num==2 :
        return 'sum_of_squares'

def check_int_input(value, min_value,max_value):
    if not value.isnumeric():
        raise NotNumberError(value)  
    if not int(value) >= min_value or not int(value) <= max_value: 
        raise ValueError

def get_int_input(string, min_value, max_value=sys.maxsize) :
    flag = False
    value = None
    while not flag :
        try :
            value = input(string)
            check_int_input(value, min_value, max_value)
            flag = True
        except ValueError: 
            print('invalid number!')
        except NotNumberError as e:
            print('invalid value, input must be a positive number!')   

    return int(value)
 

def get_configuration_net():
    print('\n\n\n')
    print('-------------------------------------NEURAL NETWORK PROJECT------------------------------------- \
          \n                            creation of a multilayer neural network\n\n\n')


    n_hidden_layers = get_int_input('define the number of hidden layers (min value = 1): ',1)

    print('\nfor each hidden layer define the number of internal nodes and the activation functions.')

    n_nodes_hidden_layers = list()  
    types_of_activation_functions_hidden_layers = list()  
    
    types_of_activation_functions()

    for i in range(n_hidden_layers): 
        print('hidden layer ',i+1,':')
    
        n_nodes=get_int_input('-  number of nodes: ',1)

        activation_function = get_int_input('-  choose activaction function: ',1, n_activation_functions)
        n_nodes_hidden_layers.append(int(n_nodes))
        types_of_activation_functions_hidden_layers.append(get_activation_function(activation_function))
        
        print('\n')
    
    print('output layer :')
    types_of_error_functions()
   
    error_function = get_int_input('-  define the error function: ',1, n_error_functions)

    return n_hidden_layers, n_nodes_hidden_layers, types_of_activation_functions_hidden_layers, get_error_function(error_function)
