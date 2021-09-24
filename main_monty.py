import utility
import numpy as np
import matplotlib.pyplot as plt
import functions as fun

from net import Net
from mnist import MNIST


def padding(feature_volume):
    remove_dim = False
    if feature_volume.ndim < 3:
        feature_volume = np.expand_dims(feature_volume, axis=0)
        remove_dim = True

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

    padded_feature_volume = np.array(padded_feature_volume)
    if remove_dim:
        padded_feature_volume = np.squeeze(padded_feature_volume, axis=0)

    return padded_feature_volume

def convolution(feature_volume, kernels, stride=1):
    feature_volume = padding(feature_volume)

    if feature_volume.ndim < 3:
        feature_volume = np.expand_dims(feature_volume, axis=0)

    n_rows = feature_volume.shape[1]
    n_columns = feature_volume.shape[2]

    # convolution
    b_index = 0
    n_kernels = kernels.shape[0]
    feature_map = list()
    feature_map_row = list()
    conv_feature_volume = list()

    for k in range(n_kernels):
        kernel = kernels[k]
        k_rows = kernel.shape[1]
        k_columns = kernel.shape[2]

        for i in range(1, n_rows - 1, stride):
            row_start = i - 1
            row_finish = row_start + k_rows

            for j in range(1, n_columns - 1, stride):
                column_start = j - 1
                column_finish = column_start + k_columns

                region = feature_volume[:, row_start:row_finish, column_start:column_finish]

                region = np.multiply(region, kernel)
                node = np.sum(region) # + bias[b_index]
                b_index += 1

                feature_map_row.append(node)

            feature_map.append(feature_map_row)
            feature_map_row[:] = []

        conv_feature_volume.append(feature_map)
        feature_map[:] = []

    return np.array(conv_feature_volume)




mndata = MNIST('./python-mnist/data')
X, t = mndata.load_training()

X = utility.get_mnist_data(X)
t = utility.get_mnist_labels(t)

x1 = X[:, 0:1].reshape(28, 28)
x2 = X[:, 1:2].reshape(28, 28)

feature_volume = list()
feature_volume.append(x1)
feature_volume.append(x2)
feature_volume = np.array(feature_volume)

kernel = np.random.uniform(size=(1, 2, 3, 3))

conv_feature_volume = convolution(feature_volume, kernel)
