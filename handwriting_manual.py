## 
import sys
import numpy as _np

from _neuralnet import NeuralNetwork
sys.path.append('__data')
from read_mnist_handwriting import *




##################
## FUNCTIONS
##
##
##


##################
##################




##################
## MAIN
##
##
if __name__=="__main__":
	
	# Final layer size:
	K = 10

	# Load data.
	print 100*'-', '\n**** LOADING MNIST-Handwriting data.'
	# Init object.
	_MN_DATA = MNIST('../__DATA/mnist/')
	# Load data from files.
	_MN_DATA.test_images, _MN_DATA.test_labels = _MN_DATA.load_testing()
	_MN_DATA.train_images, _MN_DATA.train_labels = _MN_DATA.load_training()
	# Make a selection and transpose.
	X_TRAIN = _np.asarray( _MN_DATA.train_images[:10000] ) # m-by-n
	TMP_Y = _np.asarray( _MN_DATA.train_labels[:10000] ) # m-by-1
	# Map Y into a m-by-K matrix.
	Y = _np.genfromtxt('__data-stanford/Y_NN_MLStanford.csv', delimiter=',')
	Y_TRAIN = _np.zeros(( X_TRAIN.shape[0], K ))
	for e,x in enumerate(TMP_Y): Y_TRAIN[e][x] = 1 # m-by-K
	# Show shapes (just to check)
	print '     Dataset size:', X_TRAIN.shape
	print '     Dataset size:', Y_TRAIN.shape,'\n', 100*'-'

	# Build NEURAL NETWORK
	NN = NeuralNetwork( [ X_TRAIN.shape[1],25,K ] )
	
	NN.train( 	features=X_TRAIN.transpose(), 
				classifications=Y_TRAIN.transpose(), 
				alpha=0.05, 
				lmbda=0.5, 
				maxiter=250 )



##################
##################

