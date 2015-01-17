# Libraries
import sys
import json
from datetime import datetime
import math
import time

# External Libraries
import numpy as np
import matplotlib.pyplot as plt
# import scipy.io
# import scipy.optimize

# Own Libraries
sys.path.append('')
from nn import Cost_CrossEntropy, Act_Sigmoid

sys.path.append('__data')
from mnist_load import *





##################
## FUNCTIONS & CLASSES
##
##

## Sparse AutoEncoder Class.
class SparseAutoEncoder():
	
	def __init__(self, n_input, n_hidden, rho=0.01, lamda=0.0001, beta=3.0, cost=Cost_CrossEntropy, activation=Act_Sigmoid ):
		self.__version__ = 1.5
		
		self.n_input = n_input
		self.n_hidden = n_hidden
		
		self.cost = cost
		self.activation = activation

		''' Autoencoder Parameters
			Rho: the desired average activation of hidden units.
			Lambda: weight decay parameter
			Beta: weight of sparsity penalty term.
		'''
		self.rho = rho
		self.lamda = lamda
		self.beta = beta
		
		# Set the weights and biases.
		# self.biases = [ np.random.randn(y, 1) for y in sizes[1:] ]
		# self.weights = [ np.random.randn(y,x)/np.sqrt(x) for x, y in zip(sizes[:-1], sizes[1:]) ]
		self.weights = [ self.init_weights(n_hidden, n_input), self.init_weights(n_input, n_hidden) ]
		self.biases = [ np.zeros((n_hidden, 1)), np.zeros((n_input, 1)) ]

		self.print_autoencoder()
		##################

	def init_weights( self, dim1, dim2 ):
		# return np.asarray( np.random.uniform(
		# 		low=-4 * np.sqrt(6. / (dim2 + dim1)),
		# 		high=4 * np.sqrt(6. / (dim2 + dim1)),
		# 		size=(dim1, dim2) 
		# ) )
		return np.random.randn(dim1,dim2) / np.sqrt(dim2)
		##################

	def randomise_data( self, X ):
		rnd = np.random.permutation( X.shape[1] )
		return X[:,rnd]
		##################

	def testtestest( self, X_in, epochs=2, batch_size=100 ):
		
		m = X_in.shape[1]
		W1 = self.weights[0]
		W2 = self.weights[1]
		b1 = self.biases[0]
		b2 = self.biases[1]

		# For each epoch.
		for e in range(epochs):
			start_e = datetime.now()
			# Randomize data
			X_in = self.randomise_data( X=X_in )
			J_cost = 0.0
			# Run over batches.
			for i in xrange(0, m, batch_size):
				
				lim_upper = min(m,i+batch_size)
				mb = lim_upper - i
				X = X_in[:,i:lim_upper]

				hidden_layer = self.activation.fn( zeta=np.dot(W1,X)+b1 )
				output_layer = self.activation.fn( zeta=np.dot(W2,hidden_layer)+b2 )

				""" Estimate the average activation value of the hidden layers """
				rho_cap = ( np.sum(hidden_layer, axis=1) / m ).reshape((hidden_layer.shape[0],))

				""" Compute intermediate difference values using Backpropagation algorithm """
				diff = output_layer - X
				
				sum_of_squares_error = 0.5 * np.sum( np.multiply(diff, diff) ) / m
				weight_decay         = 0.5 * self.lamda * ( np.sum( np.multiply(W1, W1) ) + np.sum( np.multiply(W2, W2) ) )
				KL_divergence        = self.beta * np.sum( self.rho * np.log(self.rho/rho_cap) + (1 - self.rho) * np.log( (1-self.rho)/(1-rho_cap) ) )
				cost                 = sum_of_squares_error + weight_decay + KL_divergence
				J_cost += cost
				if e == 0 and i == 0:
					print '\tInitial Cost: {}'.format(cost/batch_size)
				else:
					self.print_epoch_progress(epoch=e+1, pct=100*float(i+batch_size)/m )
				
				KL_div_grad = self.beta * (-(self.rho / rho_cap) + ((1 - self.rho) / (1 - rho_cap)))
				
				del_out = np.multiply( diff, np.multiply(output_layer, 1-output_layer) )
				del_hid = np.multiply(np.dot(np.transpose(W2), del_out) + np.transpose(np.matrix(KL_div_grad)), 
										 np.multiply(hidden_layer, 1-hidden_layer))

				""" Compute the gradient values by averaging partial derivatives
					Partial derivatives are averaged over all training examples """
					
				W1_grad = np.dot(del_hid, np.transpose(X))
				W2_grad = np.dot(del_out, np.transpose(hidden_layer))
				b1_grad = np.sum(del_hid, axis=1)
				b2_grad = np.sum(del_out, axis=1)
					
				W1_grad = W1_grad/m + self.lamda * W1
				W2_grad = W2_grad/m + self.lamda * W2
				b1_grad = b1_grad/m
				b2_grad = (b2_grad/m).reshape(b2.shape)

				# Update W1
				alpha = 0.5
				# assert W1.shape == W1_grad.shape
				# assert W2.shape == W2_grad.shape
				W1 = W1 - alpha * ( W1_grad + self.lamda * W1 )
				W2 = W2 - alpha * ( W2_grad + self.lamda * W2 )
				b1 = b1 - alpha * b1_grad
				b2 = b2 - alpha * b2_grad

				W1_grad, W2_grad, b1_grad, b2_grad = None, None, None, None
				del_out, del_hid = None, None
				KL_div_grad = None
			
			print '\tCost: {}'.format( J_cost/m )

		# Update...
		self.weights = [ W1, W2 ]
		self.biases = [ b1, b2 ]
		# Visualize.
		self.visualizeW1()
		##################

	def print_autoencoder(self):
		# Update.
		print 100 * '-', '\n\tAUTOENCODER INITIALIZED'
		print '\tActivation function:       ', self.activation.name
		print '\tCost function:             ', self.cost.name
		for e,w in enumerate(self.weights): print '\tWeights dimensions (',e,'):  ', w.shape
		for e,b in enumerate(self.biases): print '\tBiases dimensions (',e,'):   ', b.shape
		p = 0
		for e,w in enumerate(self.weights): p += w.shape[0] * (w.shape[1]+1)
		print '\tNumber of parameters:      ', p ,'\n', 100 * '-'
		##################

	def print_epoch_progress(self, epoch, pct):
		pct *= 72.0/100.0
		sys.stdout.write('\r')
		sys.stdout.write('    --- Epoch %d: [%-72s] %7.2f%%' % ( epoch, '*' * int(pct), 100*pct/72 ) )
		sys.stdout.flush()
		if pct==72.0: print '\r'
		##################

	def visualizeW1( self ):
		dim = int(math.sqrt(self.n_input))
		fig, ax = plt.subplots( nrows=10, ncols=10 )
		index = 0							  
		for axis in ax.flat:
			image = axis.imshow(self.weights[0][index,:].reshape(dim, dim), cmap=plt.cm.gray, interpolation='nearest' )
			axis.set_frame_on(False)
			axis.set_axis_off()
			index += 1
		plt.show()
		##################
	
	##################

##################
##################




##################
## MAIN
##
##
if __name__=="__main__":


	print 100 * '-', '\n\tLOADING MNIST-Handwriting data.'
	start = datetime.now()
	# Init object.
	_MN_DATA = MNIST('../_DATA/mnist/')
	# Load data from files.
	_MN_DATA.test_images, _MN_DATA.test_labels = _MN_DATA.load_testing()
	_MN_DATA.train_images, _MN_DATA.train_labels = _MN_DATA.load_training()
	X_max, X_min = 255, 0 # np.max(_MN_DATA.train_images) * 1.0, np.min(_MN_DATA.train_images) * 1.0
	print '\tData loaded in:             {} seconds.'.format( datetime.now()-start )
	# Make a selection, normalize and transpose.
	m, m_cv = 20000, 1000
	X_train = np.asarray( np.asarray(_MN_DATA.train_images[:m])/(X_max-X_min) ).transpose().reshape((28,28,m)) # m-by-n to n-by-m
	# X_test = np.asarray( np.asarray(_MN_DATA.train_images[m:m+m_cv])/(X_max-X_min) ).transpose()
	print '\tTraining-set shape:        ', X_train.shape
	# print '\tTesting-set shape:         ', X_test.shape, '\n', 100 * '-'


	# Create Patches.
	patch_side = 8
	num_patches = 40000
	dataset = np.zeros((patch_side*patch_side, num_patches))
	rand = np.random.RandomState(int(time.time()))
	image_indices = rand.randint(28-patch_side, size=(num_patches,2) ) # 10000-by-2 array of random numbers between 0-504(512-8)
	image_number  = rand.randint(20000, size=num_patches ) # 10000-by-1 with integer between 0-9
	for i in xrange(num_patches):
		index1 = image_indices[i, 0]
		index2 = image_indices[i, 1]
		index3 = image_number[i]
		patch = X_train[index1:index1+patch_side, index2:index2+patch_side, index3] # extract 8-by-8 patches from random image.
		patch = patch.flatten() # make it a (8x8)64-by-1 
		dataset[:, i] = patch

	# Add noise.
	# X_train = np.random.binomial( n=1, p=0.1, size=X_train.shape ) * X_train
	
	AE = SparseAutoEncoder( n_input=dataset.shape[0], n_hidden=10*10 )
	AE.testtestest( X_in=dataset, epochs=500, batch_size=50 )

	

##################
##################
