import sys

sys.path.append('../') # theano folder path.
import datasets.mnist


datasets = datasets.mnist.fn_T_load_data_MNIST('../../../_DATA/mnist.pkl.gz')
train_set_x, train_set_y = datasets[0] # <class 'theano.tensor.sharedvar.TensorSharedVariable'>
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]
image_dimensions = datasets[3]

print 100 * '-', '\n\t\tDATA LOADED!'