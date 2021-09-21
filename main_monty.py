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
def padding(feature_volume):
    depth = feature_volume.shape[0]
    rows = feature_volume.shape[1]
    columns = feature_volume.shape[2]

    feature_map = None
    padded_feature_volume = list()
    for d in range(depth):
        feature_map = feature_volume[d, :, :]

        n_columns = feature_map.shape[1]
        vzeros = np.zeros(n_columns)

        feature_map = np.vstack((feature_map, vzeros))
        feature_map = np.vstack((vzeros, feature_map))

        n_rows = feature_map.shape[0]
        hzeros = np.zeros((n_rows, 1))

        feature_map = np.hstack((feature_map, hzeros))
        feature_map = np.hstack((hzeros, feature_map))

        padded_feature_volume.append(feature_map)

    return np.array(padded_feature_volume)


def convolution(feature_volume, kernels, stride=1):
    feature_volume = padding(feature_volume)

    depth = feature_volume.shape[0]
    n_rows = feature_volume.shape[1]
    n_columns = feature_volume.shape[2]

    kernel_depth =  kernels.shape[0]
    kernel_rows = kernels.shape[1]
    kernel_columns = kernels.shape[2]

    # convolution
    results = list()
    matrix_sum = None

    conv_feature_volume = list()
    feature_map_temp = list()
    row_temp = list()

    for k in range(kernel_depth):
        kernel = kernels[k, :, :]
        for r in range(1, n_rows - 1, stride):
            for c in range(1, n_columns - 1, stride):
                row_start = r - 1
                column_start = c - 1

                row_finish = row_start + kernel_rows
                column_finish = column_start + kernel_columns

                for d in range(depth):
                    region = feature_volume[d, row_start:row_finish, column_start:column_finish]
                    matrix_prod = np.multiply(region, kernel) # + bias
                    if d == 0:
                        matrix_sum = matrix_prod
                    else:
                        matrix_sum = matrix_sum + matrix_prod

                result = np.sum(matrix_sum)
                row_temp.append(result.copy())

            feature_map_temp.append(row_temp.copy())
            row_temp[:] = []

        conv_feature_volume.append(feature_map_temp.copy())
        feature_map_temp[:] = []

    return np.array(conv_feature_volume)


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

<<<<<<< HEAD

x1 = X[:, 0:1]
x1 = x1.reshape(28, 28)
x2 = X[:, 1:2]
x2 = x2.reshape(28, 28)
x3 = X[:, 2:3]
x3 = x3.reshape(28, 28)
=======
x = X[:, 4:5]
x = x.reshape(28, 28)
>>>>>>> f5683a6ee67035ec7d698c30522ccf4e8ecc58b4

feature_volume = list()
feature_volume.append(x1)
feature_volume.append(x2)
feature_volume.append(x3)

feature_volume = np.array(feature_volume)


kernel = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
kernel_volume = list()
kernel_volume.append(kernel)
kernel_volume.append(kernel)
kernel_volume.append(kernel)

kernel_volume = np.array(kernel_volume)
conv_feature_volume = convolution(feature_volume, kernel_volume)

print(feature_volume.shape)
print(conv_feature_volume.shape)


# conv_x = convolution(x, kernel)
# pooled_x = max_pooling(conv_x, (3, 3))
#
#
# fig = plt.figure(figsize=(6, 6))
#
# subplot1 = fig.add_subplot(221)
# subplot1.title.set_text('Original image')
# plt.imshow(x, cmap='Greys')
#
# subplot2 = fig.add_subplot(222)
# subplot2.title.set_text('Convolutional image')
# plt.imshow(conv_x,  cmap='Greys')
#
# subplot3 = fig.add_subplot(223)
# subplot3.title.set_text('Pooled image')
# plt.imshow(pooled_x,  cmap='Greys')
#
# plt.show()
