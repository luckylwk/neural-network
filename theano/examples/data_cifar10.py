import sys

sys.path.append('../') # theano folder path.
import datasets.cifar10


datasets = datasets.cifar10.fn_theano_load_cifar10( 
	path_to_dir='../../../../../DATA/cifar-10-batches-py/' 
)

train_set_x, train_set_y = datasets[0] # <class 'theano.tensor.sharedvar.TensorSharedVariable'>
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]
image_dimensions = datasets[3]
datasets = None

print 100 * '-', '\n\t\tDATA LOADED!'