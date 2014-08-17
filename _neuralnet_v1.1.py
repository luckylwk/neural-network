import sys
import json

import numpy as _np
import matplotlib.pyplot as _plt

sys.path.append('__data')
from mnist_load import *


'''
Implemented
	L2 regularization (squared)

To implement
	DropOut
	DropConnect
	max-norm regularization
	L1 regularization (norm)
'''


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
	return 1.0 / (1.0 + _np.exp(-zeta))
# Vectorised return for sigmoid.
fn_sigmoid_vec = _np.vectorize(fn_sigmoid)


## Function to calculate sigmoid prime (first derivative of sigmoid).
def fn_sigmoid_prime(zeta):
	return sigmoid(zeta) * ( 1 - sigmoid(zeta) )
	##################


## Neural network class.
class NeuralNetwork():
	'''
	Data requirements in this class:
		data:   n-features by m-samples.
		labels: k-classifications by m-samples.
	'''

	def __init__( self, sizes ):
		self.version = 1.1
		self.layers = sizes
		# Weights (Theta) is a matrix with L-1 theta matrices, initialized using gaussian random.
		self.biases = [ _np.random.randn(y, 1) for y in sizes[1:] ]
		self.weights = [ _np.random.randn(y,x)/_np.sqrt(x) for x, y in zip(sizes[:-1], sizes[1:]) ]
		# Activations and gradients.
		self.activations, self.zetas, self.gradients, self.accuracy = [], [], [], []
		# Update.
		print 100 * '-', '\n    *** NEURAL NETWORK INITIALIZED'
		for e,w in enumerate(self.weights): print '        Weights dimensions (',e,'):', w.shape
		print 100 * '-'
		for e,b in enumerate(self.biases): print '        Biases dimensions (',e,'): ', b.shape
		print 100 * '-'
		##################

	def forward_propagation( self, a, store=False ):
		# Assumes X to be n-features by m-samples.
		if store: self.activations.append(a)
		for b, w in zip(self.biases, self.weights):
			if store:
				zeta = _np.dot(w, a) + b
				self.zetas.append(zeta)
				a = fn_sigmoid_vec(zeta)
				self.activations.append(a)
			else:
				a = fn_sigmoid_vec( _np.dot(w, a) + b )
		return a
		##################

	def calc_payoff( self, matrix ):
		return _np.argmax(matrix, axis=0) # Returns index. Alt: _np.amax(matrix, axis=1)
		##################

	def calc_cost( self, X, Y, lmbda=0.0, ret_acc=False ):
		a = self.forward_propagation( X ) # input is n-by-m >>> returns k-by-m
		cost = _np.nan_to_num( _np.sum( -Y*_np.log(a)-(1-Y)*_np.log(1-a) ) )
		cost += 0.5 * (lmbda/X.shape[1]) * sum(_np.linalg.norm(w)**2 for w in self.weights)
		if ret_acc:
			return cost, self.calc_accuracy( X=X, Y=Y, calc=False, a=a )
		else:
			return cost
		##################

	def calc_accuracy(self, X, Y, calc=True, a=None ):
		pred = _np.argmax( self.forward_propagation( X ), axis=0 ) if calc else _np.argmax( a, axis=0 )
		real = _np.argmax( Y, axis=0 )
		return 100 * float((pred==real).sum()) / Y.shape[1]
		##################


	## TRAINING SECTION.

	def back_propagation(self, X, Y):
		# X and Y are assumed to be n-by-mb and k-by-mb matrices representing a mb-sized batch of m.
		# Returns the gradients for each bias and weights, initialize with zeros.
		grad_b = [ _np.zeros(b.shape) for b in self.biases ]
		grad_w = [ _np.zeros(w.shape) for w in self.weights ]
		# Forward propogation to get activations and zeta's.
		self.forward_propagation( a=X, store=True )
		# Calculate and populate the last layers.
		delta = self.activations[-1] - Y # k-by-mb
		# Update last gradient-layers.
		grad_b[-1] = delta.sum(axis=1).reshape((delta.shape[0],1)) # sum over columns to k-by-1
		grad_w[-1] = _np.dot(delta, self.activations[-2].transpose()) # k-by-mb times mb-by-someL - k-by-someL
		# Update the remaining gradient-layers
		for l in xrange(2, len(self.layers)):
			# print self.zetas[-l].shape # someL-by-mb
			spv = fn_sigmoid(self.zetas[-l])*(1-fn_sigmoid(self.zetas[-l]))
			# print spv.shape # someL-by-mb
			delta = _np.dot( self.weights[-l+1].transpose(), delta ) * spv
			# print delta.shape # someL-by-mb
			grad_b[-l] = delta.sum(axis=1).reshape((delta.shape[0],1)) # sum over columns to someL-by-1
			grad_w[-l] = _np.dot(delta, self.activations[-l-1].transpose())
		# Update gradients.
		self.gradients = [ grad_b, grad_w ]
		##################

	def randomise_data(self,X,Y):
		rnd = _np.random.permutation( X.shape[1] )
		return X[:,rnd], Y[:,rnd]
		##################

	def stochastic_gradient_descent(self, X, Y, X_CV=None, Y_CV=None, epochs=10, batch_size=10, eta=0.5, lmbda=0.0 ):
		print 100 * '-', '\n    *** STARTING TRAINING (Stochastic Gradient Descent)\n'
		m = X.shape[1]
		for e in xrange(epochs):
			X, Y = self.randomise_data(X=X, Y=Y)
			for i in xrange(0, m, batch_size):
				self.print_epoch_progress(epoch=e, pct=100*float(i+batch_size)/m )
				# Perform batch-backpropagation to obtain the gradients.
				self.activations, self.zetas, self.gradients = [], [], []
				lim_upper = min(m,i+batch_size)
				mb = lim_upper - i
				self.back_propagation(X=X[:,i:lim_upper],Y=Y[:,i:lim_upper])
				# Update the biases and weights using the gradients.
				self.biases = [ b-(eta/mb)*gb for b, gb in zip(self.biases, self.gradients[0]) ]
				self.weights = [ (1-eta*(lmbda/m))*w-(eta/mb)*gw for w, gw in zip(self.weights, self.gradients[1]) ]
			# Use the weights to get the results.
			cost_train, acc_train = self.calc_cost( X=X_train, Y=Y_train, lmbda=lmbda, ret_acc=True )
			#acc_train = self.calc_accuracy( X=X_train, Y=Y_train )
			self.accuracy.append(acc_train)
			print '        Training-cost:       ', cost_train
			print '        Training-accuracy:   {}%'.format( acc_train )
			if X_CV is not None:
				cost_cv, acc_cv = self.calc_cost( X=X_CV, Y=Y_CV, lmbda=lmbda, ret_acc=True )
				print '        Validation-cost:     ', cost_cv
				print '        Validation-accuracy:  {}%\n'.format( acc_cv )
		# Training completed. Plot graph?
		_plt.plot( _np.arange(0,epochs), self.accuracy )
		_plt.show()
		##################

	def print_epoch_progress(self, epoch, pct):
		pct *= 72.0/100.0
		sys.stdout.write('\r')
		sys.stdout.write('        Epoch %d: [%-72s] %7.2f%%' % ( epoch, '*' * int(pct), 100*pct/72 ) )
		sys.stdout.flush()
		if pct==72.0: print '\r'
		##################


	## FILE HANDLING.

	def save_to_file(self, X, Y, PATH, FILE):
		output = {
			"sizes": self.layers,
			"weights": [ w.tolist() for w in self.weights ],
			"biases": [ b.tolist() for b in self.biases ],
			"cost": self.calc_cost( X=X, Y=Y, lmbda=0.0 ),
			"accuracy": self.calc_accuracy( X=X, Y=Y )
		}
		print '        Saving to file ', FILE
		f = open(PATH + FILE, "w")
		json.dump(obj=output, fp=f, indent=4)
		f.close()
	##################

##################
##################




##################
## MAIN
##
##
if __name__=="__main__":


	print 100 * '-', '\n    *** LOADING MNIST-Handwriting data.'
	# Init object.
	_MN_DATA = MNIST('../_DATA/mnist/')
	# Load data from files.
	_MN_DATA.test_images, _MN_DATA.test_labels = _MN_DATA.load_testing()
	_MN_DATA.train_images, _MN_DATA.train_labels = _MN_DATA.load_training()
	X_max, X_min = _np.max(_MN_DATA.train_images)*1.0, _np.min(_MN_DATA.train_images)*1.0
	# Make a selection, normalize and transpose.
	m = 5000
	X_train = _np.asarray( _MN_DATA.train_images[:m]/(X_max-X_min) ).transpose() # m-by-n to n-by-m
	Y_train = _np.asarray( [ fn_vectorize(_) for _ in _MN_DATA.train_labels[:m] ] ).reshape((m,10)).transpose() # m-by-1 to k-by-m
	X_test = _np.asarray( _MN_DATA.train_images[m:m+m]/(X_max-X_min) ).transpose()
	Y_test = _np.asarray( [ fn_vectorize(_) for _ in _MN_DATA.train_labels[m:m+m] ] ).reshape((m,10)).transpose()
	print '        Training-set shape: ', X_train.shape, '\n        Testing-set shape:  ', Y_train.shape, '\n', 100 * '-'


	##################
	# Build NEURAL NETWORK
	NN = NeuralNetwork( sizes=[ X_train.shape[0],30,10 ] )
	print '        Initial cost:     ', NN.calc_cost( X=X_train, Y=Y_train )
	print '        Initial accuracy: ', NN.calc_accuracy( X=X_train, Y=Y_train )
	
	NN.stochastic_gradient_descent( X=X_train, Y=Y_train, 
		X_CV=X_test, Y_CV=Y_test, 
		epochs=20, batch_size=10, 
		eta=0.5, lmbda=0.0 )

	print '        Saving NeuralNetwork parameters to file.'
	NN.save_to_file( X=X_train, Y=Y_train, PATH='', FILE='test_save.json' )

##################
##################
