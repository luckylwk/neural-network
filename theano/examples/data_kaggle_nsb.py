import sys

sys.path.append('../') # theano folder path.
import datasets.kaggle


datasets = datasets.kaggle.fn_theano_load_kaggle_national_science_bowl( 
	path_to_file='../../../../../DATA/kaggle-national-data-science-bowl/train-pickle.p' 
)

train_set_x, train_set_y = datasets[0] # <class 'theano.tensor.sharedvar.TensorSharedVariable'>
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]
image_dimensions = datasets[3]
datasets = None

print 100 * '-', '\n\t\tDATA LOADED!'