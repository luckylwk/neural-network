# THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python run_mlp.py

import sys
from collections import OrderedDict


import numpy as np
import theano


sys.path.append('../') # theano folder path.

import datasets.mnist
from layers.activation import ReLU, SoftMax
from models.mlp import MLP
from trainers import GradientDescent

random_seed = 1234




if __name__ == '__main__':


	# Load the dataset.
	datasets = datasets.mnist.fn_T_load_data_MNIST('../../../_DATA/mnist.pkl.gz')
	# train_set_x, train_set_y = datasets[0] # <class 'theano.tensor.sharedvar.TensorSharedVariable'>
	# valid_set_x, valid_set_y = datasets[1]
	# test_set_x, test_set_y = datasets[2]
	# image_dimensions = datasets[3]


	# BUILD the MODEL.
	rng = np.random.RandomState(random_seed)
	dropout = True
	x = theano.tensor.matrix('x') # <class 'theano.tensor.var.TensorVariable'>
	y = theano.tensor.ivector('y')

	# Create the MODEL.
	__MODEL__ = MLP(
		rng=rng, 
		init_input=x,
		layer_sizes=[ 28*28, 1200, 1200, 10 ],
		dropout_rates=[ 0.2, 0.2, 0.2 ], # rate is the chance something is dropped.
		activations=[ ReLU, ReLU, SoftMax ],
		use_bias=True 
	)
	# # Build the expresson for the cost function.
	# cost = __MODEL__.negative_log_likelihood(y)
	# dropout_cost = __MODEL__.dropout_negative_log_likelihood(y)



	# TRAINING SETUP.
	initial_learning_rate = 1.0
	learning_rate_decay = 0.998
	batch_size = 100
	__TRAINER__ = GradientDescent( 
		datasets=datasets,
		X=x, y=y,
		model=__MODEL__
	)
	__TRAINER__.train( 
		batch_size=batch_size, 
		init_learning_rate=initial_learning_rate, 
		learning_rate_decay=learning_rate_decay 
	)
	
	# n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
	# n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
	# n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size


	
	
	# # the params for momentum
	# momentum_start = 0.5
	# momentum_end = 0.99
	# # for epoch in [0, momentum_epoch_interval], the momentum increases linearly
	# # from mom_start to mom_end. After momentum_epoch_interval, it stay at mom_end
	# momentum_epoch_interval = 500
	# squared_filter_length_limit = 15.0

	# index = theano.tensor.lscalar()
	# epoch = theano.tensor.scalar()
	# learning_rate = theano.shared( np.asarray( initial_learning_rate, dtype=theano.config.floatX ) )


	# # Compute the GRADIENTS of the MODEL with respect to the PARAMS
	# gradients = []
	# for param in __MODEL__.params:
	# 	# Use the right cost function here to train with or without dropout.
	# 	gradients.append( theano.tensor.grad( dropout_cost if dropout else cost, param ) )

	# # ... and allocate memory for momentum'd versions of the GRADIENTS
	# gradients_momentum = []
	# for param in __MODEL__.params:
	# 	gparam_mom = theano.shared( np.zeros( param.get_value(borrow=True).shape, dtype=theano.config.floatX) )
	# 	gradients_momentum.append(gparam_mom)

	# # Compute momentum for the current epoch
	# mom = ifelse( 
	# 	epoch < momentum_epoch_interval,
	# 	momentum_start * (1.0 - epoch/momentum_epoch_interval) + momentum_end * (epoch/momentum_epoch_interval),
	# 	momentum_end
	# )


	# # Update the step direction using momentum
	# updates = OrderedDict()
	# for grad_mom, grad in zip( gradients_momentum, gradients ):
	# 	updates[grad_mom] = mom * grad_mom - (1. - mom) * learning_rate * grad

	# # ... and take a step along that direction
	# for param, gparam_mom in zip( __MODEL__.params, gradients_momentum ):
	# 	stepped_param = param + updates[gparam_mom]
	# 	# This is a silly hack to constrain the norms of the rows of the weight
	# 	# matrices.  This just checks if there are two dimensions to the
	# 	# parameter and constrains it if so... maybe this is a bit silly but it
	# 	# should work for now.
	# 	if param.get_value(borrow=True).ndim == 2:
	# 		#squared_norms = T.sum(stepped_param**2, axis=1).reshape((stepped_param.shape[0],1))
	# 		#scale = T.clip(T.sqrt(squared_filter_length_limit / squared_norms), 0., 1.)
	# 		#updates[param] = stepped_param * scale
			
	# 		# constrain the norms of the COLUMNs of the weight, according to
	# 		# https://github.com/BVLC/caffe/issues/109
	# 		col_norms = theano.tensor.sqrt( theano.tensor.sum( theano.tensor.sqr(stepped_param), axis=0 ) )
	# 		desired_norms = theano.tensor.clip( col_norms, 0, theano.tensor.sqrt(squared_filter_length_limit) )
	# 		scale = desired_norms / ( 1e-7 + col_norms )
	# 		updates[param] = stepped_param * scale
	# 	else:
	# 		updates[param] = stepped_param



	# # Compile theano function for training.  This returns the training cost and
	# # updates the model parameters.
	# output = dropout_cost if dropout else cost
	# __TRAIN_MODEL__ = theano.function(
	# 	inputs=[epoch, index], 
	# 	outputs=output,
	# 	updates=updates,
	# 	givens={
	# 		x: train_set_x[index * batch_size:(index + 1) * batch_size],
	# 		y: train_set_y[index * batch_size:(index + 1) * batch_size]
	# 	}
	# )

	# __CV_MODEL__ = theano.function(
	# 	inputs=[index],
 #        outputs=[ __MODEL__.negative_log_likelihood(y), __MODEL__.errors(y) ],
 #        givens={
 #            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
 #            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
 #        }
 #    )
	# __TEST_MODEL__ = theano.function(
	# 	inputs=[index],
 #        outputs=__MODEL__.errors(y),
 #        givens={
 #            x: test_set_x[index * batch_size:(index + 1) * batch_size],
 #            y: test_set_y[index * batch_size:(index + 1) * batch_size]
 #        }
 #    )

	# # Theano function to decay the learning rate, this is separate from the
	# # training function because we only want to do this once each epoch instead
	# # of after each minibatch.
	# decay_learning_rate = theano.function(
	# 	inputs=[], 
	# 	outputs=learning_rate,
	# 	updates={ learning_rate: learning_rate * learning_rate_decay } 
	# )

	# # EPOCHS
	
	# for e in range(5):

	# 	# Minibatches. -- not stochastic!!?
	# 	train_cost = 0.0
	# 	for minibatch_index in xrange(n_train_batches):
	# 		train_cost += __TRAIN_MODEL__( e+1, minibatch_index )
	# 		progress_pct = 100.0 * (minibatch_index+1)/n_train_batches
	# 		if progress_pct % 5.0 == 0:
	# 			fn_print_epoch_progress( epoch=e+1, pct=progress_pct, cost=train_cost/(minibatch_index+1) )

	# 	# Cross-validation.
	# 	cv_cost, cv_error = 0.0, 0.0
	# 	for i in xrange(n_valid_batches):
	# 		c, e = __CV_MODEL__(i)
	# 		cv_cost += c
	# 		cv_error += 100.0 * e

	# 	# cv_cost, cv_errors = np.mean( [ __CV_MODEL__(i) for i in xrange(n_valid_batches) ] ) * 100.0
	# 	# print this_validation_errors

	# 	print '\t\tCV Cost: %7.3f --- CV Error: %5.3f%%' % ( cv_cost/n_valid_batches, cv_error/n_valid_batches )
	# 	# Update the learning rate using the defined Theano function.
	# 	new_learning_rate = decay_learning_rate()












