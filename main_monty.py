import utility
import numpy as np
import matplotlib.pyplot as plt


from net import Net
from mnist import MNIST
from learning import batch_learning, standard_gradient_descent, back_propagation


def convolution(image):
    kernel = [[-1, -1, -1], [-1, 2, -1], [-1, -1, -1]]
    kernel = np.array(kernel)

    n_rows = image.shape[0]
    n_columns = image.shape[1]

    new_image = list()
    temp_result = list()

    for x in range(1, n_rows - 1):
        for y in range(1, n_columns - 1):
            row_start = x - 1
            row_finish = row_start + (kernel.shape[0])
            column_start = y - 1
            column_finish = column_start + (kernel.shape[1])

            region = image[row_start:row_finish, column_start:column_finish]

            result = np.multiply(region, kernel)
            result = np.sum(result)

            temp_result.append(result)

        new_image.append(temp_result.copy())
        temp_result[:] = []

    return np.array(new_image)


# caricamento dataset
mndata = MNIST('./python-mnist/data')
X, t = mndata.load_training()

'''
n_hidden_layers, \
n_hidden_nodes_per_layer, \
act_fun_codes, \
error_fun_code = utility.get_configuration_net()

# creazione rete
net = Net(n_hidden_layers=n_hidden_layers,
          n_hidden_nodes_per_layer=n_hidden_nodes_per_layer,
          act_fun_codes=act_fun_codes,
          error_fun_code=error_fun_code)

net.print_config()
'''

net = Net(n_hidden_layers=1,
          n_hidden_nodes_per_layer=[4],
          act_fun_codes=[0, 1],
          error_fun_code=1)

net.print_config()

X = utility.get_mnist_data(X)
t = utility.get_mnist_labels(t)

X_train = X[:, 0:20]
t_train = t[:, 0:20]

X_val = X[:, 21:26]
t_val = t[:, 21:26]


# best_net = batch_learning(net, X_train, t_train, X_val, t_val)

weights_deriv, bias_deriv = back_propagation(net, X_train, t_train)
# net = standard_gradient_descent(net, weights_deriv, bias_deriv, 0.1)

layer_weights_deriv = weights_deriv[0]
print(layer_weights_deriv.shape)

# y_train = net.sim(X_train)
# y_val = net.sim(X_val)


# --------------CONVOLUZIONE------------------
'''
x = image.reshape(28, 28)
conv_x = convolution(x)

fig = plt.figure(figsize=(1, 2))
fig.add_subplot(1, 2, 1)
plt.imshow(x)

fig.add_subplot(1, 2, 2)
plt.imshow(conv_x)

plt.show()
'''
