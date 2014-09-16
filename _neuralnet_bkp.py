# Libraries
import sys
import json
from datetime import datetime

# External Libraries
import numpy as _np
import matplotlib.pyplot as _plt

# Own Libraries
from _library import Cost_CrossEntropy, Act_Sigmoid
sys.path.append('__data')
from mnist_load import *



'''
Implemented
	L2 regularization (squared)

To implement
	DropOut - Bernoulli and Gaussian?
	DropConnect
	max-norm regularization
	L1 regularization (norm)

	Hinton (2014): 
		'Although dropout alone gives significant improvements, using 
		dropout along with max- norm regularization, large decaying 
		learning rates and high momentum provides a significant boost 
		over just using dropout'

	Decoupled cost/activation functions?

	http://stackoverflow.com/questions/24351206/backpropagation-for-rectified-linear-unit-activation-with-cross-entropy-error
	http://en.wikipedia.org/wiki/Hinge_loss
	http://www.datarobot.com/blog/regularized-linear-regression-with-scikit-learn/
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


# ## Function to calculate sigmoid.
# def fn_sigmoid(zeta):
# 	return 1.0 / (1.0 + _np.exp(-zeta))
# # Vectorised return for sigmoid.
# fn_sigmoid_vec = _np.vectorize(fn_sigmoid)


# ## Function to calculate sigmoid prime (first derivative of sigmoid).
# def fn_sigmoid_prime(zeta):
# 	return sigmoid(zeta) * ( 1 - sigmoid(zeta) )
# 	##################


## Neural network class.
class NeuralNetwork():
	'''
	Data requirements in this class:
		data:   n-features by m-samples.
		labels: k-classifications by m-samples.
	'''

	def __init__( self, sizes, cost=Cost_CrossEntropy, activation=Act_Sigmoid ):
		self.version = 1.2
		self.layers = sizes
		self.cost = cost
		self.activation = activation
		# Weights (Theta) is a matrix with L-1 theta matrices, initialized using gaussian random.
		self.biases = [ _np.random.randn(y, 1) for y in sizes[1:] ]
		self.weights = [ _np.random.randn(y,x)/_np.sqrt(x) for x, y in zip(sizes[:-1], sizes[1:]) ]
		# Activations and gradients.
		self.activations, self.zetas, self.gradients, self.acc, self.acc_cv = [], [], [], [], []
		# Update.
		print 100 * '-', '\n    *** NEURAL NETWORK INITIALIZED'
		for e,w in enumerate(self.weights): print '        Weights dimensions (',e,'):', w.shape
		print 100 * '-'
		for e,b in enumerate(self.biases): print '        Biases dimensions (',e,'): ', b.shape
		print 100 * '-'
		##################

	def forward_propagation( self, a, r=None, store=False ):
		# Assumes X to be n-features by m-samples.
		if r is None: r = self.generate_dropouts( mb=a.shape[1], binomials=False )
		if store: self.activations.append(a)
		for b, w, rp in zip(self.biases, self.weights, r):
			a *= rp # dropout activations.
			if store:
				zeta = _np.dot(w, a) + b
				self.zetas.append(zeta)
				#a = fn_sigmoid_vec(zeta)
				a = self.activation.fn( zeta=zeta, vectorize=True )
				self.activations.append(a)
			else:
				#a = fn_sigmoid_vec( _np.dot(w, a) + b )
				a = self.activation.fn( zeta=_np.dot(w, a)+b, vectorize=True )
		return a
		##################

	def calc_payoff( self, matrix ):
		return _np.argmax(matrix, axis=0) # Returns index. Alt: _np.amax(matrix, axis=1)
		##################

	def calc_cost( self, X, Y, lmbda=0.0, ret_acc=False ):
		a = self.forward_propagation( X ) # input is n-by-m >>> returns k-by-m
		#cost = _np.nan_to_num( _np.sum( -Y*_np.log(a)-(1-Y)*_np.log(1-a) ) ) / X.shape[1]
		cost = self.cost.fn( activations=a, Y=Y ) / X.shape[1]
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

	def back_propagation(self, X, Y, r):
		# X and Y are assumed to be n-by-mb and k-by-mb matrices representing a mb-sized batch of m.
		# Returns the gradients for each bias and weights, initialize with zeros.
		grad_b = [ _np.zeros(b.shape) for b in self.biases ]
		grad_w = [ _np.zeros(w.shape) for w in self.weights ]
		# Forward propogation to get activations and zeta's.
		self.forward_propagation( a=X, r=r, store=True )
		# Calculate and populate the last layers.
		#delta = self.activations[-1] - Y # k-by-mb
		delta = self.cost.delta( zeta=self.zetas[-1], activations=self.activations[-1], Y=Y ) # k-by-mb
		# Update last gradient-layers.
		grad_b[-1] = delta.sum(axis=1).reshape((delta.shape[0],1)) # sum over columns to k-by-1
		grad_w[-1] = _np.dot(delta, self.activations[-2].transpose()) # k-by-mb times mb-by-someL - k-by-someL
		# Update the remaining gradient-layers
		for l in xrange(2, len(self.layers)):
			spv = fn_sigmoid(self.zetas[-l])*(1-fn_sigmoid(self.zetas[-l]))
			delta = _np.dot( self.weights[-l+1].transpose(), delta ) * spv
			grad_b[-l] = delta.sum(axis=1).reshape((delta.shape[0],1)) # sum over columns to someL-by-1
			grad_w[-l] = _np.dot(delta, self.activations[-l-1].transpose())
		# Update gradients.
		self.gradients = [ grad_b, grad_w ]
		##################

	def randomise_data(self,X,Y):
		rnd = _np.random.permutation( X.shape[1] )
		return X[:,rnd], Y[:,rnd]
		##################

	def generate_dropouts(self, mb, binomials=True):
		# Dropout - generate binomials.
		r = [ _np.ones((self.layers[0],mb)) ]
		if binomials:
			r += [ _np.random.binomial(1,0.5,(l,mb)) * 1.0 for l in self.layers[1:-1] ]
		else:
			r += [ _np.ones((l,mb)) for l in self.layers[1:-1] ]
		r += [ _np.ones((self.layers[-1],mb)) ]
		return r
		##################

	def stochastic_gradient_descent(self, X, Y, X_CV=None, Y_CV=None, 
		epochs=10, batch_size=10, eta=0.5, lmbda=0.0, dropout=False ):
		print 100 * '-', '\n    *** STARTING TRAINING (Stochastic Gradient Descent)'
		m = X.shape[1]
		cost_train, acc_train = self.calc_cost( X=X_train, Y=Y_train, lmbda=lmbda, ret_acc=True )
		self.acc.append(acc_train), self.acc_cv.append(acc_train)
		print '        Initial cost:          ', cost_train
		print '        Initial accuracy:       {}%\n'.format( acc_train )
		# Start training using batch-stochastic-gradient-descent.
		for e in xrange(epochs):
			start_e = datetime.now()
			X, Y = self.randomise_data(X=X, Y=Y)
			for i in xrange(0, m, batch_size):
				self.print_epoch_progress(epoch=e, pct=100*float(i+batch_size)/m )
				# Housekeeping.
				self.activations, self.zetas, self.gradients = [], [], []
				lim_upper = min(m,i+batch_size)
				mb = lim_upper - i
				# Dropout - generate binomials.
				r = self.generate_dropouts( mb=mb ) if dropout else self.generate_dropouts( mb=mb, binomials=False )
				# Perform batch-backpropagation to obtain the gradients.
				self.back_propagation(X=X[:,i:lim_upper], Y=Y[:,i:lim_upper], r=r)
				# Update the biases and weights using the gradients.
				self.biases = [ b-(eta/mb)*gb for b, gb in zip(self.biases, self.gradients[0]) ]
				self.weights = [ (1-eta*(lmbda/m))*w-(eta/mb)*gw for w, gw in zip(self.weights, self.gradients[1]) ]
			# Use the weights to get the results.
			cost_train, acc_train = self.calc_cost( X=X_train, Y=Y_train, lmbda=lmbda, ret_acc=True )
			self.acc.append(acc_train)
			print '        Training-cost:         ', cost_train
			print '        Training-accuracy:      {}%'.format( acc_train)
			print '        Accuracy improvement:   {}%'.format( acc_train-self.acc[-2] )
			if X_CV is not None:
				cost_cv, acc_cv = self.calc_cost( X=X_CV, Y=Y_CV, lmbda=lmbda, ret_acc=True )
				self.acc_cv.append(acc_cv)
				print '        Validation-cost:       ', cost_cv
				print '        Validation-accuracy:    {}%'.format( acc_cv )
			print '        Epoch time elapsed:     {} seconds.\n'.format( datetime.now()-start_e )
		# Done. Plot accuracy graphs.
		self.plot_accuracy()
		##################

	def print_epoch_progress(self, epoch, pct):
		pct *= 72.0/100.0
		sys.stdout.write('\r')
		sys.stdout.write('    --- Epoch %d: [%-72s] %7.2f%%' % ( epoch, '*' * int(pct), 100*pct/72 ) )
		sys.stdout.flush()
		if pct==72.0: print '\r'
		##################

	def plot_accuracy(self):
		_plt.plot( _np.arange(0,len(self.acc)), self.acc, 'g' )
		if len(self.acc_cv)>1: _plt.plot( _np.arange(0,len(self.acc)), self.acc_cv, 'r' )
		_plt.ylim([0,100])
		_plt.show()
		##################

	## FILE HANDLING.

	def save_to_file(self, X, Y, PATH, FILE):
		cost, acc = self.calc_cost( X=X, Y=Y, lmbda=0.0, ret_acc=True )
		# Create output
		output = {
			"sizes": self.layers,
			"weights": [ w.tolist() for w in self.weights ],
			"biases": [ b.tolist() for b in self.biases ],
			"cost": cost,
			"accuracy": acc
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
	start = datetime.now()
	# Init object.
	_MN_DATA = MNIST('../_DATA/mnist/')
	# Load data from files.
	_MN_DATA.test_images, _MN_DATA.test_labels = _MN_DATA.load_testing()
	_MN_DATA.train_images, _MN_DATA.train_labels = _MN_DATA.load_training()
	X_max, X_min = _np.max(_MN_DATA.train_images) * 1.0, _np.min(_MN_DATA.train_images) * 1.0
	print '        Data loaded in:      {} seconds.'.format( datetime.now()-start )
	# Make a selection, normalize and transpose.
	m, m_cv = 5000, 1000
	X_train = _np.asarray( _MN_DATA.train_images[:m]/(X_max-X_min) ).transpose() # m-by-n to n-by-m
	Y_train = _np.asarray( [ fn_vectorize(_) for _ in _MN_DATA.train_labels[:m] ] ).reshape((m,10)).transpose() # m-by-1 to k-by-m
	X_test = _np.asarray( _MN_DATA.train_images[m:m+m_cv]/(X_max-X_min) ).transpose()
	Y_test = _np.asarray( [ fn_vectorize(_) for _ in _MN_DATA.train_labels[m:m+m_cv] ] ).reshape((m_cv,10)).transpose()
	print '        Training-set shape: ', X_train.shape, '\n        Testing-set shape:  ', Y_train.shape, '\n', 100 * '-'


	##################
	# Build NEURAL NETWORK
	NN = NeuralNetwork( sizes=[ X_train.shape[0],50,30,10 ] )
	
	NN.stochastic_gradient_descent( X=X_train, Y=Y_train, X_CV=X_test, Y_CV=Y_test, 
		epochs=10, batch_size=10, eta=0.6, lmbda=0.05, dropout=False )

	# print '        Saving NeuralNetwork parameters to file.'
	# NN.save_to_file( X=X_train, Y=Y_train, PATH='', FILE='test_save.json' )

##################
##################
