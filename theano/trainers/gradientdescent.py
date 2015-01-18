from collections import OrderedDict


import numpy as np
import theano
from theano.ifelse import ifelse


from trainers import fn_print_epoch_progress



class GradientDescent(object):

	def __init__( self, datasets, X, y, model ):
		
		print 100 * '-', '\n\t    --- Starting TRAINING'

		self.datasets = datasets

		self.X = X
		self.y = y

		self.model = model
		self.cost = model.negative_log_likelihood(y)
		self.dropout_cost = model.dropout_negative_log_likelihood(y)

		self.epoch = theano.tensor.scalar()

		self.gradients = []
		self.gradients_momentum = []
		self.updates = OrderedDict()
		##################

	def setup_parameters( self ):
		
		print '\t\t\tSetting up Training parameters and environment'
		dropout = True
		
		model = self.model

		for param in model.params:
			# Use the right cost function here to train with or without dropout.
			self.gradients.append( theano.tensor.grad( self.dropout_cost if dropout else self.cost, param ) )

		# ... and allocate memory for momentum'd versions of the GRADIENTS
		for param in model.params:
			gparam_mom = theano.shared( np.zeros( param.get_value(borrow=True).shape, dtype=theano.config.floatX) )
			self.gradients_momentum.append(gparam_mom)

		# the params for momentum
		momentum_start = 0.5
		momentum_end = 0.99
		# for epoch in [0, momentum_epoch_interval], the momentum increases linearly
		# from mom_start to mom_end. After momentum_epoch_interval, it stay at mom_end
		momentum_epoch_interval = 500
		# Compute momentum for the current epoch
		self.mom = ifelse( 
			self.epoch < momentum_epoch_interval,
			momentum_start * (1.0 - self.epoch/momentum_epoch_interval) + momentum_end * (self.epoch/momentum_epoch_interval),
			momentum_end
		)

		# Update the step direction using momentum
		for grad_mom, grad in zip( self.gradients_momentum, self.gradients ):
			self.updates[grad_mom] = self.mom * grad_mom - (1. - self.mom) * self.learning_rate * grad

		# ... and take a step along that direction
		squared_filter_length_limit = 15.0
		for param, gparam_mom in zip( model.params, self.gradients_momentum ):
			stepped_param = param + self.updates[gparam_mom]
			# This is a silly hack to constrain the norms of the rows of the weight
			# matrices.  This just checks if there are two dimensions to the
			# parameter and constrains it if so... maybe this is a bit silly but it
			# should work for now.
			if param.get_value(borrow=True).ndim == 2:
				#squared_norms = T.sum(stepped_param**2, axis=1).reshape((stepped_param.shape[0],1))
				#scale = T.clip(T.sqrt(squared_filter_length_limit / squared_norms), 0., 1.)
				#updates[param] = stepped_param * scale
				
				# constrain the norms of the COLUMNs of the weight, according to
				# https://github.com/BVLC/caffe/issues/109
				col_norms = theano.tensor.sqrt( theano.tensor.sum( theano.tensor.sqr(stepped_param), axis=0 ) )
				desired_norms = theano.tensor.clip( col_norms, 0, theano.tensor.sqrt(squared_filter_length_limit) )
				scale = desired_norms / ( 1e-7 + col_norms )
				self.updates[param] = stepped_param * scale
			else:
				self.updates[param] = stepped_param
		##################

	def train( self, batch_size, init_learning_rate, learning_rate_decay ):
		
		self.learning_rate = theano.shared( np.asarray( init_learning_rate, dtype=theano.config.floatX ) )
		self.setup_parameters()
		print '\t\t\tStarting training...'

		# Deal with the data.
		datasets = self.datasets
		train_set_x, train_set_y = datasets[0] # <class 'theano.tensor.sharedvar.TensorSharedVariable'>
		valid_set_x, valid_set_y = datasets[1]
		test_set_x, test_set_y = datasets[2]
		image_dimensions = datasets[3]
		
		n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
		n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
		n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

		index = theano.tensor.lscalar()
		x = self.X
		y = self.y

		# Compile theano function for training.  This returns the training cost and
		# updates the model parameters.
		dropout = True
		output = self.dropout_cost if dropout else self.cost
		__TRAIN_MODEL__ = theano.function(
			inputs=[ self.epoch, index ], 
			outputs=output,
			updates=self.updates,
			givens={
				x: train_set_x[index * batch_size:(index + 1) * batch_size],
				y: train_set_y[index * batch_size:(index + 1) * batch_size]
			}
		)

		__CV_MODEL__ = theano.function(
			inputs=[ index ],
	        outputs=[ self.model.negative_log_likelihood(y), self.model.errors(y) ],
	        givens={
	            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
	            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
	        }
	    )
		__TEST_MODEL__ = theano.function(
			inputs=[ index ],
	        outputs=self.model.errors(y),
	        givens={
	            x: test_set_x[index * batch_size:(index + 1) * batch_size],
	            y: test_set_y[index * batch_size:(index + 1) * batch_size]
	        }
	    )

		# Theano function to decay the learning rate, this is separate from the
		# training function because we only want to do this once each epoch instead
		# of after each minibatch.
		decay_learning_rate = theano.function(
			inputs=[], 
			outputs=self.learning_rate,
			updates={ self.learning_rate: self.learning_rate * learning_rate_decay } 
		)



		for e in range(5):

			# Minibatches. -- not stochastic!!?
			train_cost = 0.0
			for minibatch_index in xrange(n_train_batches):
				train_cost += __TRAIN_MODEL__( e+1, minibatch_index )
				progress_pct = 100.0 * (minibatch_index+1)/n_train_batches
				if progress_pct % 5.0 == 0:
					fn_print_epoch_progress( epoch=e+1, pct=progress_pct, cost=train_cost/(minibatch_index+1) )

			# Cross-validation.
			cv_cost, cv_error = 0.0, 0.0
			for i in xrange(n_valid_batches):
				c, e = __CV_MODEL__(i)
				cv_cost += c
				cv_error += 100.0 * e

			# cv_cost, cv_errors = np.mean( [ __CV_MODEL__(i) for i in xrange(n_valid_batches) ] ) * 100.0
			# print this_validation_errors

			print '\t\tCV Cost: %7.3f --- CV Error: %5.3f%%' % ( cv_cost/n_valid_batches, cv_error/n_valid_batches )
			# Update the learning rate using the defined Theano function.
			new_learning_rate = decay_learning_rate()

		##################

	##################