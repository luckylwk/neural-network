from collections import OrderedDict


import numpy as np
import theano





class GradientDescent(object):

	def __init__( self, datasets, model, batch_size, learning_rate, learning_rate_decay ):
		
		print 100 * '-', '\n\t    --- Starting TRAINING'

		train_set_x, train_set_y = datasets[0] # <class 'theano.tensor.sharedvar.TensorSharedVariable'>
		valid_set_x, valid_set_y = datasets[1]
		test_set_x, test_set_y = datasets[2]
		image_dimensions = datasets[3]
		n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
		n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
		n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

		self.__MODEL__ = model

		self.gradients = []
		self.gradients_momentum = []

		self.updates = OrderedDict()

		self.index = theano.tensor.lscalar()
		self.epoch = theano.tensor.scalar()
		##################

	@classmethod
	def setup_parameters( self ):
		dropout = True

		print '\t\tSetting up Training parameters and environment'
		
		for param in self.__MODEL__.params:
			# Use the right cost function here to train with or without dropout.
			self.gradients.append( theano.tensor.grad( dropout_cost if dropout else cost, param ) )
		##################

	@classmethod
	def train( self ):
		print 'training'
		##################

	##################