import os
import sys
import time
import json

# Import external libraries.
import numpy as np
import theano
import theano.tensor.signal.downsample

# Import custom modules.
from _load_data import fn_T_load_data_CIFAR10, fn_T_load_data_CIFAR10_PRED
#from logistic_sgd import LogisticRegression



##################
## FUNCTIONS & CLASSES
##
##
def fn_print_epoch_progress( epoch, pct ):
	sys.stdout.write('\r')
	sys.stdout.write('    --- Epoch %d: [%-50s] %7.2f%%' % ( epoch, '*' * int(pct), 2.0*pct ) )
	sys.stdout.flush()
	if pct==50.0: print '\r'
	##################


class ConvLayer(object):
	
	def __init__( self, rng, input, filter_shape, image_shape, poolsize=(2, 2), W_in=None, b_in=None ):
		# input = 4D Tensor <class 'theano.tensor.var.TensorVariable'>
		# filter_shape = 4-var <type 'tuple'> - #kernels, layers?, kernel dim (2)
		# image_shape = 4-var <type 'tuple'>
		print '    --- Initialising Convolutional Layer'
		assert image_shape[1] == filter_shape[1]
		print '        Input size:  {}-by-{}'.format( image_shape[2],image_shape[2] )
		print '        Kernel size: {}-by-{} | downsampled using max-pooling'.format( filter_shape[2],filter_shape[2] )
		print '        Output size: {}'.format( (image_shape[2]-filter_shape[2]+1)/poolsize[0] )

		self.input = input

		# there are "num input feature maps * filter height * filter width" inputs to each hidden unit
		fan_in = np.prod(filter_shape[1:])
		# each unit in the lower layer receives a gradient from:
		# "num output feature maps * filter height * filter width" / pooling size
		# each output map is dim-by-dim / pool. and there are x kernels.
		fan_out = filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(poolsize)

		# initialize weights with random weights
		W_bound = np.sqrt( 6. / (fan_in + fan_out) )
		W_values = rng.uniform( low=-W_bound, high=W_bound, size=filter_shape ) if W_in is None else W_in
		self.W = theano.shared( 
			value=np.asarray( W_values, dtype=theano.config.floatX ),
			borrow=True 
		)
		print '        Weights:     {}'.format( self.W.shape.eval() ) # self.W.get_value() for values.
		# Biases
		b_values = np.zeros( (filter_shape[0],), dtype=theano.config.floatX ) if b_in is None else np.asarray( b_in, dtype=theano.config.floatX )
		self.b = theano.shared( value=b_values, borrow=True )
		print '        Biases:      {}'.format( self.b.shape.eval() )

		# convolve input feature maps with filters
		# Docs: http://deeplearning.net/software/theano/library/tensor/nnet/conv.html#theano.tensor.nnet.conv.conv2d
		# <class 'theano.tensor.var.TensorVariable'>
		conv_out = theano.tensor.nnet.conv.conv2d( 
			input=input, 
			filters=self.W, 
			filter_shape=filter_shape, 
			image_shape=image_shape, 
			border_mode='valid' 
		)

		# downsample each feature map individually, using maxpooling
		# http://deeplearning.net/software/theano/library/tensor/signal/downsample.html
		# <class 'theano.tensor.var.TensorVariable'>
		pooled_out = theano.tensor.signal.downsample.max_pool_2d( input=conv_out, ds=poolsize, ignore_border=True )

		# add the bias term. Since the bias is a vector (1D array), we first reshape it to a tensor of 
		# shape (1,n_filters,1,1). Each bias will thus be broadcasted across mini-batches and feature map width & height
		# self.output = theano.tensor.tanh( pooled_out + self.b.dimshuffle('x',0,'x','x') )
		# self.output = theano.tensor.nnet.softplus( pooled_out + self.b.dimshuffle('x',0,'x','x') )
		self.output = theano.tensor.nnet.sigmoid( pooled_out + self.b.dimshuffle('x',0,'x','x') )

		# store parameters of this layer
		self.params = [ self.W, self.b ]
		##################


class HiddenLayer(object):

	def __init__( self, rng, input, n_in, n_out, W_in=None, b_in=None, activation=theano.tensor.nnet.sigmoid ):
		
		print '    --- Initialising Hidden Layer'
		print '        Input size:  {}'.format( n_in )
		print '        Output size: {}'.format( n_out )
		print '        Activation function ', type(activation)

		self.input = input
		
		# Weights
		W_bound = np.sqrt(6. / (n_in + n_out))
		if W_in is None:
			W_values = np.asarray( rng.uniform( low=-W_bound, high=W_bound, size=(n_in, n_out) ), dtype=theano.config.floatX )
			if activation == theano.tensor.nnet.sigmoid:
				W_values *= 4.0
		else:
			W_values = np.asarray( W_in, dtype=theano.config.floatX )
		self.W = theano.shared( value=W_values, name='W', borrow=True )

		# Biases
		if b_in is None:
			b_values = np.zeros( (n_out,), dtype=theano.config.floatX )
		else:
			b_values = np.asarray( b_in, dtype=theano.config.floatX )
		self.b = theano.shared( value=b_values, name='b', borrow=True )

		lin_output = theano.tensor.dot(input, self.W) + self.b
		
		self.output = ( lin_output if activation is None else activation(lin_output) )
		
		self.params = [ self.W, self.b ]
		##################


class LogisticRegression(object):
	
	def __init__( self, input, n_in, n_out, W=None, b=None ):
		
		print '    --- Initialising LOGISTIC REGRESSION Output Layer'
		print '        Input size:  {}'.format( n_in )
		print '        Output size: {}'.format( n_out )

		# initialize with 0 the weights W as a matrix of shape (n_in, n_out)
		if W is None:
			W_values = np.zeros( (n_in, n_out), dtype=theano.config.floatX )
		else:
			W_values = np.asarray( W, dtype=theano.config.floatX )

		self.W = theano.shared( value=W_values, name='W', borrow=True )
		
		# initialize the baises b as a vector of n_out 0s
		if b is None:
			b_values = np.zeros( (n_out,), dtype=theano.config.floatX )
		else:
			b_values = np.asarray( b, dtype=theano.config.floatX )
		
		self.b = theano.shared( value=b_values, name='b', borrow=True )

		# compute vector of class-membership probabilities in symbolic form
		self.p_y_given_x = theano.tensor.nnet.softmax( theano.tensor.dot(input, self.W) + self.b )

		# compute prediction as class whose probability is maximal in symbolic form
		self.y_pred = theano.tensor.argmax( self.p_y_given_x, axis=1 )

		# parameters of the model
		self.params = [ self.W, self.b ]
		##################

	def negative_log_likelihood(self, y):
		return -theano.tensor.mean( theano.tensor.log(self.p_y_given_x)[ theano.tensor.arange(y.shape[0]), y ] )
		##################

	def errors(self, y):
		if y.ndim != self.y_pred.ndim:
			raise TypeError('y should have the same shape as self.y_pred',
				('y', target.type, 'y_pred', self.y_pred.type))
		# check if y is of the correct datatype
		if y.dtype.startswith('int'):
			# the T.neq operator returns a vector of 0s and 1s, where 1 represents a mistake in prediction
			return theano.tensor.mean(theano.tensor.neq(self.y_pred, y))
		else:
			raise NotImplementedError()
		##################

	##################


##################
##################




##################
## MAIN
##
##
if __name__=="__main__":


	# SETUP
	learning_rate = 0.4
	n_epochs = 50
	pooling_dimension = 2
	kernels = [ (64,5,5,1), (128,13,13,1) ] # 2 layers, 1st: K kernels, 5x5 with X stride.
	batch_size = 100
	rng = np.random.RandomState(42)
	_LOAD_FILE_ = ''
	_SAVE_FILE_ = '_parameters/param__convnet_cifar_rgb_64x128x256x10___XXpct.json'
	_TYPE_ = 'TRAIN'

	if _TYPE_ == 'TRAIN':
		# LOAD DATA - CIFAR10
		datasets = fn_T_load_data_CIFAR10()
		train_set_x, train_set_y = datasets[0] # <class 'theano.tensor.sharedvar.TensorSharedVariable'>
		valid_set_x, valid_set_y = datasets[1]
		test_set_x, test_set_y = datasets[2]
		image_dimensions = datasets[3]
		datasets = None
		# Compute number of minibatches for training, validation and testing
		n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
		n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
		n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size
	else:
		num_pred = 300000
		pred_batch = 10000
		predict_set_x = theano.shared( fn_T_load_data_CIFAR10_PRED( range(1,pred_batch+1) ), borrow=True )
		image_dimensions = (32,32,1)
		pred_epochs = num_pred / pred_batch

	# Allocate symbolic variables for the data
	index = theano.tensor.lscalar() # index to a [mini]batch 
	x = theano.tensor.matrix('x')   # <class 'theano.tensor.var.TensorVariable'>
	y = theano.tensor.ivector('y')  # <class 'theano.tensor.var.TensorVariable'>



	######################
	# BUILD ACTUAL MODEL #
	######################

	print 100*'-', '\n    *** Building model: CONVOLUTIONAL NEURAL NETWORK'

	# Loading parameters...
	_PARAMS_LOADED_ = dict()
	if _LOAD_FILE_ != '':
		print '        Loading parameters from file'
		_PARAMS_LOADED_ = json.load( open( _LOAD_FILE_, "r+" ) )

	# Reshape matrix of rasterized images to a 4D tensor
	# <class 'theano.tensor.var.TensorVariable'>
	_LAYER_0_ = x.reshape( (batch_size, image_dimensions[2], image_dimensions[0], image_dimensions[1]) )


	# Construct the first convolutional pooling layer.
	tmp_W, tmp_b = None, None
	if 'conv1_W' in _PARAMS_LOADED_:
		tmp_W = _PARAMS_LOADED_['conv1_W']
		tmp_b = _PARAMS_LOADED_['conv1_b']
	_LAYER_1_ = ConvLayer( 
		rng=rng, 
		input=_LAYER_0_,
		image_shape=( batch_size, image_dimensions[2], image_dimensions[0], image_dimensions[1] ),
		filter_shape=( kernels[0][0], image_dimensions[2], kernels[0][1], kernels[0][2] ), 
		poolsize=( pooling_dimension, pooling_dimension ),
		W_in=tmp_W, b_in=tmp_b
	)


	# Construct the second convolutional pooling layer
	size_l1 = ( image_dimensions[0] - kernels[0][1] + 1 ) / pooling_dimension
	tmp_W, tmp_b = None, None
	if 'conv2_W' in _PARAMS_LOADED_:
		tmp_W = _PARAMS_LOADED_['conv2_W']
		tmp_b = _PARAMS_LOADED_['conv2_b']
	_LAYER_2_ = ConvLayer( 
		rng=rng, 
		input=_LAYER_1_.output,
		image_shape=( batch_size, kernels[0][0], size_l1, size_l1 ),
		filter_shape=( kernels[1][0], kernels[0][0], kernels[1][1], kernels[1][2] ), 
		poolsize=( pooling_dimension, pooling_dimension ),
		W_in=tmp_W, b_in=tmp_b
	)
	

	# construct a fully-connected sigmoidal layer
	size_l2 = ( size_l1 - kernels[1][1] + 1 ) / pooling_dimension
	tmp_W, tmp_b = None, None
	if 'hidden3_W' in _PARAMS_LOADED_:
		tmp_W = _PARAMS_LOADED_['hidden3_W']
		tmp_b = _PARAMS_LOADED_['hidden3_b']
	_LAYER_3_ = HiddenLayer( 
		rng=rng, 
		input=_LAYER_2_.output.flatten(2), 
		n_in=kernels[1][0] * size_l2 * size_l2,
		n_out=256, 
		activation=theano.tensor.nnet.sigmoid,
		W_in=tmp_W, b_in=tmp_b
	)


	# Setup the OUTPUT layer.
	tmp_W, tmp_b = None, None
	if 'output_W' in _PARAMS_LOADED_:
		tmp_W = _PARAMS_LOADED_['output_W']
		tmp_b = _PARAMS_LOADED_['output_b']
	_LAYER_4_ = LogisticRegression( input=_LAYER_3_.output, n_in=256, n_out=10, W=tmp_W, b=tmp_b )


	# the cost we minimize during training is the NLL of the model
	_COST_ = _LAYER_4_.negative_log_likelihood(y)


	if _TYPE_ == 'TRAIN':

		# create a function to compute the mistakes that are made by the model
		_MODEL_TEST_ = theano.function(
			inputs=[index], 
			outputs=_LAYER_4_.errors(y),
			givens={
				x: test_set_x[index * batch_size : (index + 1) * batch_size],
				y: test_set_y[index * batch_size : (index + 1) * batch_size]
			}
		)

		_MODEL_CV_ = theano.function(
			inputs=[index], 
			outputs=_LAYER_4_.errors(y),
			givens={
				x: valid_set_x[index * batch_size: (index + 1) * batch_size],
				y: valid_set_y[index * batch_size: (index + 1) * batch_size]
			}
		)

		# create a list of all model parameters to be fit by gradient descent
		_PARAMS_ = _LAYER_1_.params + _LAYER_2_.params + _LAYER_3_.params + _LAYER_4_.params

		# create a list of gradients for all model parameters
		_GRADIENTS_ = theano.tensor.grad( _COST_, _PARAMS_ )

		# train_model is a function that updates the model parameters by SGD Since this model has many parameters, 
		# it would be tedious to manually create an update rule for each model parameter. We thus
		# create the updates list by automatically looping over all (params[i],grads[i]) pairs.
		updates = []
		for param_i, grad_i in zip( _PARAMS_, _GRADIENTS_ ):
			updates.append( (param_i, param_i - learning_rate * grad_i) )

		_MODEL_TRAIN_ = theano.function(
			inputs=[index], 
			outputs=_COST_, 
			updates=updates,
			givens={
				x: train_set_x[ index * batch_size : (index + 1) * batch_size ],
				y: train_set_y[ index * batch_size : (index + 1) * batch_size ]
			}
		)

		_MODEL_TRAIN_EVAL_ = theano.function(
			inputs=[index], 
			outputs=_LAYER_4_.errors(y),
			givens={
				x: train_set_x[ index * batch_size : (index + 1) * batch_size ],
				y: train_set_y[ index * batch_size : (index + 1) * batch_size ]
			}
		)


		###############
		# TRAIN MODEL #
		###############

		print 100 * '-', '\n    *** Model TRAINING Initialised.'
		
		# 
		start_time = time.clock()

		epoch = 0
		done_looping = False

		# For each epoch.
		while ( epoch < n_epochs ) and ( not done_looping ):
			epoch = epoch + 1
			epoch_start_time = time.clock()
			# For each minibatch.
			for minibatch_index in xrange(n_train_batches):
				# Print progress.
				fn_print_epoch_progress( epoch=epoch, pct=50*float(minibatch_index+1)/n_train_batches )
				# Train the model
				cost_ij = _MODEL_TRAIN_( minibatch_index )

			training_eval = _MODEL_TRAIN_EVAL_( minibatch_index )
			print '                 Training time: {} seconds'.format( time.clock()-epoch_start_time )
			print '                 Cost:          {}'.format( cost_ij )
			print '                 Training-error:{}'.format( np.mean(training_eval) * 100.0 )
			# compute zero-one loss on validation set
			validation_losses = [ _MODEL_CV_(i) for i in xrange(n_valid_batches) ]
			print '                 CV-error:      {}'.format( np.mean(validation_losses) * 100.0 )
			# test it on the test set
			test_losses = [ _MODEL_TEST_(i) for i in xrange(n_test_batches) ]
			print '                 Test-error:    {}'.format( np.mean(test_losses) * 100.0 )
			

		print 100*'-', '\n    *** TOTAL TRAINING TIME: ', time.clock() - start_time
		


		###################
		# SAVE PARAMETERS #
		###################

		print '    *** Saving Parameters.'

		_OUTPUT_ = dict()
		_OUTPUT_ = { 
			"conv1_W": [ w.tolist() for w in _LAYER_1_.W.get_value() ],
			"conv1_b": _LAYER_1_.b.get_value().tolist(),
			"conv2_W": [ w.tolist() for w in _LAYER_2_.W.get_value() ],
			"conv2_b": _LAYER_2_.b.get_value().tolist(),
			"hidden3_W": [ w.tolist() for w in _LAYER_3_.W.get_value() ],
			"hidden3_b": _LAYER_3_.b.get_value().tolist(),
			"output_W": [ w.tolist() for w in _LAYER_4_.W.get_value() ],
			"output_b": _LAYER_4_.b.get_value().tolist()
		}
		
		# Create output
		f = open( _SAVE_FILE_, "w" )
		json.dump( obj=_OUTPUT_, fp=f, indent=4 )
		f.close()

	elif _TYPE_ == 'TEST':

		###########
		# PREDICT #
		###########

		print 100 * '-', '\n    *** Model PREDICTION Started'

		categories = {	0:'frog',1:'truck',2:'deer',3:'automobile',4:'bird',5:'horse', 6:'ship',7:'cat',8:'dog',9:'airplane' }
		# create a function to compute the mistakes that are made by the model
		_MODEL_PREDICT_ = theano.function(
			inputs=[index], 
			outputs=_LAYER_4_.y_pred,
			givens={
				x: predict_set_x[index * batch_size : (index + 1) * batch_size]
			}
		)

		# Open output file
		image_nr = 0
		f_out = open( '_submissions/submission.csv', 'w+' )
		f_out.write( 'id,label' + os.linesep )
		for pe in range(0,pred_epochs): #pred_epochs
			if pe != 0:
				# load new data.
				predict_set_x.set_value( fn_T_load_data_CIFAR10_PRED( range(pred_batch*pe+1,pred_batch*(pe+1)+1) ) )
			# run each batch
			for pb in range(0,pred_batch/batch_size):
				test = _MODEL_PREDICT_( pb )
				#print test
				for each in test:
					image_nr += 1
					f_out.write( str(image_nr) + ',' + categories[int(each)] + os.linesep )

		f_out.close()
		

##################
##################

