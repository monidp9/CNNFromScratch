import utility
import numpy as np
import matplotlib.pyplot as plt

from net import Net
from mnist import MNIST


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




mndata = MNIST('./python-mnist/data')
images, labels = mndata.load_training()

# utility.get_configuration_net()



image = images[0]
image = np.array(image)

'''
TEST CONVOLUZIONE
x = image.reshape(28, 28)
conv_x = convolution(x)

fig = plt.figure(figsize=(1, 2))
fig.add_subplot(1, 2, 1)
plt.imshow(x)

fig.add_subplot(1, 2, 2)
plt.imshow(conv_x)

plt.show()
'''


def identity(x):
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))



# creazione rete
net = Net(n_hidden_layers=1,
          n_hidden_nodes=[3],
          activation_functions=[sigmoid, identity],
          error_function=[identity])

image = image.reshape(-1, 1)
# input, output = net.forwardStep(image)

x = np.array([2000, 3000, -2000])
print('x', x)
print('sig', sigmoid(x))
