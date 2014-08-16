##  
##
import sys

import numpy as _np

sys.path.append('__data')
from mnist_load import *



##################
## FUNCTIONS & CLASSES
##
##

## Miscellaneous functions
def fn_vectorize(d):
	v = _np.zeros((10, 1))
	v[d] = 1.0
	return v
	##################


## Function to calculate sigmoid.
def fn_sigmoid(zeta):
	return 1.0 / (1.0 + _np.exp(-zeta) )
	##################
fn_sigmoid_vec = _np.vectorize(fn_sigmoid)


## Function to calculate sigmoid prime (first derivative of sigmoid).
def fn_sigmoid_prime(zeta):
	return sigmoid(zeta) * ( 1 - sigmoid(zeta) )
	##################


## Neural network class.
class NeuralNetwork():
	
	def __init__( self, sizes ):
		self.version = 1.0
		# Number of neuron-layers (L) is the length of the vector.
		self.num_L = len(sizes) # Includes the input and output.
		# L with L(i)
		self.layers = sizes
		# Weights (Theta) is a matrix with L-1 theta matrices, initialized using gaussian random.
		self.weights = _np.array([ 
				_np.asarray( _np.random.randn(y,(x+1)) ) 
				for x, y in zip( sizes[:-1], sizes[1:] ) 
			])
		# Activations and gradients.
		self.activations, self.gradients = [], []
		print 100*'-', '\n**** NEURAL NETWORK INITIALIZED'
		for e,w in enumerate(self.weights): print '     Weights dimensions (',e,'):', w.shape
		print 100*'-'
		##################

	def forward_propagation( self, input0, payoff=True, training=False ):
		# For each step (1-2, 2-3, 3-4). L-1 steps
		matrix = input0.transpose() if input0.shape[0] != self.layers[0] else input0
		for w in self.weights:
			matrix = fn_sigmoid( _np.dot( w,_np.insert(matrix,0,1,axis=0) ) ) # Add bias 1.
			if training: self.activations.append( matrix )
		# Done > use payoff function and return result.
		return _np.ravel( self.calc_payoff( matrix ) ) if payoff==True else matrix
		##################

	def calc_payoff( self, matrix ):
		return _np.argmax(matrix, axis=0) # Returns index. Alt: _np.amax(matrix, axis=1)
		##################

	def calc_cost( self, h, y, m, lmbda ):
		assert h.shape == y.shape # Should both be K x m. With K the size of the output layer L
		cost_std = _np.sum( _np.nan_to_num( -y*_np.log(h) - (1-y)*_np.log(1-h) ) ) / m
		cost_reg = 0
		for theta in self.weights: cost_reg += lmbda * _np.sum(theta**2) / (2*m)	
		return cost_std + cost_reg
		##################

	def train( self, features, classifications, alpha=0.5, lmbda=0.0, maxiter=200 ):
		assert features.shape[0] == self.layers[0]
		m = features.shape[1]
		print 100 * '-', '\n**** ATTEMPTING TRAINING...\n' 
		# Loop for a number of iterations
		for i in range(maxiter):
			print '     ---- Iteration', i+1, '\t',
			# First one needs self.activations to be cleaned and initialized.
			if i == 0:
				# Features: n-by-m matrix. Add ones for bias term.
				self.activations = [features] #[ _np.insert(features,0,1,axis=0) ]
				cost_prev = 0
			else:
				# Remove previous activations (but keep features)
				for _ in range(self.num_L-3): self.activations.pop()
			# Estimate the outputs for delta calculations. Activations also generated and saved.
			output_est = self.forward_propagation( input0=features, payoff=False, training=True )
			cost = self.calc_cost( h=output_est, y=classifications, m=m, lmbda=lmbda )
			print("         | Cost: {0:.8f}\t| {1:.8f} diff.\t| Accuracy: {2}%".format( 
				cost, (cost-cost_prev), self.accuracy( h=output_est, y=classifications, m=m ) ) )
			cost_prev = cost
			# Gradient Descent (using back-prop) to update the weights.
			self.grad_desc( h=output_est, y=classifications, m=m, lmbda=lmbda, alpha=alpha )
		# Done > print line.
		print '\n', 100*'-', '\n'
		##################

	def grad_desc( self, h, y, m, lmbda, alpha ):
		# Use back-propagation to fill the Gradients.
		self.back_propagation( h=h, y=y, m=m, lmbda=lmbda )
		# Update weights using the gradients.
		self.weights = [ w - alpha*g for w,g in zip(self.weights, self.gradients) ]
		##################

	def back_propagation( self, h, y, m, lmbda ):
		# Starting with the last layer.
		for l in range(self.num_L):
			if l==0:
				small_delta = _np.transpose(h-y) # m-by-K
			else:
				ix = self.num_L-1-l
				if ix > 0:
					big_delta = _np.dot( _np.insert(self.activations[ix],0,1,axis=0), small_delta ) / m
					small_delta = _np.dot( small_delta,self.weights[ix][:,1:] ) * \
						_np.transpose( self.activations[ix] * (1-self.activations[ix]) )
				else:
					big_delta = _np.dot( _np.insert(self.activations[ix],0,1,axis=0), small_delta ) / m
				# Create gradients and add.
				self.gradients.append( _np.transpose(big_delta) + lmbda * self.weights[ix] )
		# Done > reverse gradients.
		self.gradients.reverse()
		##################

	def accuracy( self, h, y, m ):
		return float( 100* (self.calc_payoff( matrix=h ) == self.calc_payoff( matrix=y )).sum() ) / m
		return 'unknown'
		##################

	##################

##################
##################




##################
## MAIN
##
##
if __name__=="__main__":
	
	
	# # Load data from CSVs
	# print 100*'-', '\n**** LOADING Stanford data to validate the NeuralNetwork'
	# Theta1 = _np.genfromtxt('__data-stanford/T1_NN_MLStanford.csv', delimiter=',')
	# Theta2 = _np.genfromtxt('__data-stanford/T2_NN_MLStanford.csv', delimiter=',')
	# X = _np.genfromtxt('__data-stanford/X_NN_MLStanford.csv', delimiter=',') # m-by-n
	# # Load Y and map it from 5000x1 to 5000x10.
	# Y = _np.genfromtxt('__data-stanford/Y_NN_MLStanford.csv', delimiter=',') # m-by-1
	# Y_mtrx = _np.zeros(( X.shape[0], 10 ))
	# for e,x in enumerate(Y): Y_mtrx[e][x-1] = 1 # m-by-K
	# # Print updates.
	# print '     Theta1 size: ', Theta1.shape
	# print '     Theta2 size: ', Theta2.shape
	# print '     Dataset size:', X.shape,'\n', 100*'-'

	# # Initialize neural network and set the weights.
	# NN = NeuralNetwork( [ X.shape[1],25,10 ] )
	# print '     Loading in Stanford weights...'
	# NN.weights[0] = Theta1
	# NN.weights[1] = Theta2
	# print '     Weights set. Dimensions', NN.weights[0].shape, NN.weights[1].shape, '\n', 100*'-'

	# # Check performance of cost function...
	# y_hat = NN.forward_propagation( input0=X.transpose(), payoff=False, training=False )
	# print '**** CHECKING Performance'
	# cost = NN.calc_cost( h=y_hat, y=Y_mtrx.transpose(), m=X.shape[0], lmbda=0 )
	# if (cost-0.287629512178)<0.000001: 
	# 	print '     Cost function check: PASSED (Cost:',cost,' vs. 0.287629512178)'  
	# else: 
	# 	print '     Cost function check: FAILURE'

	# # Train the Network.
	# NN.train( features=X.transpose(), classifications=Y_mtrx.transpose(), alpha=0.025, lmbda=0, maxiter=10 )



	print 100 * '-', '\n    *** LOADING MNIST-Handwriting data.'
	# Init object.
	_MN_DATA = MNIST('../_DATA/mnist/')
	# Load data from files.
	_MN_DATA.test_images, _MN_DATA.test_labels = _MN_DATA.load_testing()
	_MN_DATA.train_images, _MN_DATA.train_labels = _MN_DATA.load_training()
	X_max, X_min = _np.max(_MN_DATA.train_images)*1.0, _np.min(_MN_DATA.train_images)*1.0
	# Make a selection, normalize and transpose.
	m = 10000
	X_train = _np.asarray( _MN_DATA.train_images[:m]/(X_max-X_min) ).transpose() # m-by-n to n-by-m
	Y_train = _np.asarray( [ fn_vectorize(_) for _ in _MN_DATA.train_labels[:m] ] ).reshape((m,10)).transpose() # m-by-1 to k-by-m
	X_test = _np.asarray( _MN_DATA.train_images[m:m+m]/(X_max-X_min) ).transpose()
	Y_test = _np.asarray( [ fn_vectorize(_) for _ in _MN_DATA.train_labels[m:m+m] ] ).reshape((m,10)).transpose()
	print '        Training-set shape: ', X_train.shape, '\n        Testing-set shape:  ', Y_train.shape, '\n', 100 * '-'

	##################
	# Build NEURAL NETWORK
	NN = NeuralNetwork( sizes=[ X_train.shape[0],50,30,10 ] )
	NN.train( features=X_train, classifications=Y_train, maxiter=1000 )


##################
##################

