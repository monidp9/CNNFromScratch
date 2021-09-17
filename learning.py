import net
import functions as fun


def back_progagation(net, x, t):
    
    layer_input, layer_output = net.forwardStep(x)
    
    derivative_error_function = net.activation_function_derivative[i](parametro)
    print(type(net.error_function))
    print(derivative_error_function)    