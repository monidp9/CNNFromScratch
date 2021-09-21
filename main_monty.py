import utility
import numpy as np
import matplotlib.pyplot as plt
import functions as fun

from net import Net
from mnist import MNIST
from learning import batch_learning, standard_gradient_descent, back_propagation


net = Net(n_hidden_layers=1,
          n_hidden_nodes_per_layer=[100],
          act_fun_codes=[2,1],
          error_fun_code=1)

net.print_config()

# DATASET MNIST

# mndata = MNIST('./python-mnist/data')
# X, t = mndata.load_training()

# X = utility.get_mnist_data(X)
# t = utility.get_mnist_labels(t)
#
# X, t = utility.get_random_dataset(X, t, 2000)
#
# X_train, X_val, t_train, t_val = utility.train_test_split(X, t)
# net = batch_learning(net, X_train, t_train, X_val, t_val)



# DATASET IRIS
from sklearn.datasets import load_iris
dataset = load_iris()
X_train = np.transpose(dataset.data)
t_train = dataset.target

t_train = utility.get_iris_labels(t_train)

X_train, X_val, t_train, t_val = utility.train_test_split(X_train, t_train, test_size=0.20)
X_train, X_test, t_train, t_test = utility.train_test_split(X_train, t_train, test_size=0.01)

net = batch_learning(net, X_train, t_train, X_val, t_val)

x = X_test[:, 0:2]
y = net.sim(x)

print(fun.softmax(y))
print(t_test[:, 0:2])



# --------------------------------------------
# X_train = X_train[:, 0:3]
# t_train = t_train[:, 0:3]
#
# y = net.sim(X_train)













# --------------CONVOLUZIONE------------------
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
