import utility
import numpy as np
import matplotlib.pyplot as plt
import functions as fun

from net import Net
from mnist import MNIST
from learning import batch_learning, standard_gradient_descent, back_propagation


# net = Net(n_hidden_layers=1,
#           n_hidden_nodes_per_layer=[100],
#           act_fun_codes=[2, 1],
#           error_fun_code=1)
#
# net.print_config()

# DATASET MNIST

mndata = MNIST('./python-mnist/data')
X, t = mndata.load_training()

X = utility.get_mnist_data(X)
t = utility.get_mnist_labels(t)

# X, t = utility.get_random_dataset(X, t, 2000)
#
# X_train, X_val, t_train, t_val = utility.train_test_split(X, t)
# net = batch_learning(net, X_train, t_train, X_val, t_val)



# DATASET IRIS
# from sklearn.datasets import load_iris
# dataset = load_iris()
# X_train = np.transpose(dataset.data)
# t_train = dataset.target
#
# t_train = utility.get_iris_labels(t_train)
#
# X_train, X_val, t_train, t_val = utility.train_test_split(X_train, t_train, test_size=0.15)
# X_train, X_test, t_train, t_test = utility.train_test_split(X_train, t_train, test_size=0.01)
#
# net = batch_learning(net, X_train, t_train, X_val, t_val)
#
# x = X_test[:, 0:2]
# y = net.sim(x)
#
# print(fun.softmax(y))
# print(t_test[:, 0:2])




# --------------CONVOLUZIONE------------------
def convolution(image, kernel, stride=1):
    kernel = np.array(kernel)

    new_image = list()
    temp_result = list()

    # padding
    n_columns = image.shape[1]
    vzeros = np.zeros(n_columns)
    image = np.vstack((image, vzeros))
    image = np.vstack((vzeros, image))

    n_rows = image.shape[0]
    hzeros = np.zeros((n_rows, 1))
    image = np.hstack((image, hzeros))
    image = np.hstack((hzeros, image))

    n_rows = image.shape[0]
    n_columns = image.shape[1]

    # convolution
    for i in range(1, n_rows - 1, stride):
        for j in range(1, n_columns - 1, stride):
            row_start = i - 1
            column_start = j - 1

            row_finish = row_start + (kernel.shape[0])
            column_finish = column_start + (kernel.shape[1])

            region = image[row_start:row_finish, column_start:column_finish]

            result = np.multiply(region, kernel)
            result = np.sum(result)

            temp_result.append(result)

        new_image.append(temp_result.copy())
        temp_result[:] = []

    return np.array(new_image)

def max_pooling(x, region_size):
    n_rows = x.shape[0]
    n_columns = x.shape[1]

    row_stride = region_size[0]
    column_stride = region_size[1]

    pooled_x = list()
    temp_result = list()

    for i in range(1, n_rows - 1, row_stride):
        for j in range(1, n_columns - 1, column_stride):
            row_start = i - 1
            column_start = j - 1

            row_finish = row_start + row_stride
            column_finish = column_start + column_stride

            region = x[row_start:row_finish, column_start:column_finish]
            max = np.max(region)

            temp_result.append(max)

        pooled_x.append(temp_result.copy())
        temp_result[:] = []

    return np.array(pooled_x)


x = X[:, 4:5]
x = x.reshape(28, 28)

kernel = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]

conv_x = convolution(x, kernel)
pooled_x = max_pooling(conv_x, (3, 3))


fig = plt.figure(figsize=(6, 6))

subplot1 = fig.add_subplot(221)
subplot1.title.set_text('Original image')
plt.imshow(x, cmap='Greys')

subplot2 = fig.add_subplot(222)
subplot2.title.set_text('Convolutional image')
plt.imshow(conv_x,  cmap='Greys')

subplot3 = fig.add_subplot(223)
subplot3.title.set_text('Pooled image')
plt.imshow(pooled_x,  cmap='Greys')

plt.show()
