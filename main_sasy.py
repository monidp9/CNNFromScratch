from learning import back_progagation
import utility
import numpy as np
import matplotlib.pyplot as plt

from net import Net
from mnist import MNIST


# caricamento dataset
mndata = MNIST('./python-mnist/data')
images, labels = mndata.load_training()

net = Net(n_hidden_layers=1,
          n_hidden_nodes_per_layer=[3],
          act_fun_codes=[0,1],
          error_fun_code=0)

net.print_config()

image = images[0]
label = labels[0]

image = np.array([0.2, 0.5])
image=image.reshape(-1,1)
label = [1,1]

back_progagation(net,image,label)
