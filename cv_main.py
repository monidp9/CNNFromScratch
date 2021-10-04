from cv_learning import batch_learning
import utility as utl
from cv_net import ConvolutionalNet
from mnist import MNIST

# parametri di default 
n_cv_layers = 1
n_kernels_per_layer = [1]
n_hidden_nodes = 10
act_fun_codes = [1,2]
error_fun_code = 1

# caricamento dataset
mndata = MNIST('./python-mnist/data')
X, t = mndata.load_training()
X = utl.get_mnist_data(X)
t = utl.get_mnist_labels(t)

X,t = utl.get_random_dataset(X,t, n_samples = 10) 
X = utl.get_scaled_data(X)

X_train, X_test, t_train, t_test = utl.train_test_split(X, t, test_size = 0.25)
X_train, X_val, t_train, t_val = utl.train_test_split(X_train, t_train, test_size = 0.25)

if not utl.is_standard_conf(): 
    n_cv_layers, n_kernels_per_layer, n_hidden_nodes, act_fun_codes, error_fun_code = utl.get_conf_cv_net()

net = ConvolutionalNet(n_cv_layers = n_cv_layers, 
                       n_kernels_per_layer = n_kernels_per_layer,
                       n_hidden_nodes = n_hidden_nodes, 
                       act_fun_codes = act_fun_codes, 
                       error_fun_code = error_fun_code)

net.print_config()
batch_learning(net, X_train, t_train, X_val, t_val)

y_test = net.sim(X_test)

accuracy = utl.get_metric_value(y_test, t_val, 'accuracy')
precision = utl.get_metric_value(y_test, t_val, 'precision')
recall = utl.get_metric_value(y_test, t_val, 'recall')
f1 = utl.get_metric_value(y_test, t_val, 'f1')

print('     accuracy: {:.2f} - precision: {:.2f} - recall: {:.2f} - f1: {:.2f}'.format(accuracy, precision, recall, f1))