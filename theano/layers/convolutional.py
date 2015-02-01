import numpy as np
import theano
import theano.tensor.signal.downsample

from activation import *
from weights import create_weights_uniform




class ConvolutionalLayer(object):

	def __init__( self, rng, layer_input, input_dim, kernels, kernel_dim, kernel_stride, pooling, activation, W_in=None, b_in=None, verbose=True ):

		if verbose:
			print '\t\t    --- Initialising CONVOLUTIONAL Layer'
			print '\t\t\t# Kernels:           {}'.format( kernels )
			print '\t\t\tKernel Dimensions:   {}'.format( kernel_dim )
			print '\t\t\tKernel Stride:       {}'.format( kernel_stride )
			print '\t\t\tPooling:             {}'.format( pooling )
			print '\t\t\tActivation function: {}'.format( activation.name )
			print '\t\t\tInput Dimension:     {}'.format( input_dim )
			new_dim_1 = ( input_dim[2] - kernel_dim[0] + 1 ) / pooling[0]
			new_dim_2 = ( input_dim[3] - kernel_dim[1] + 1 ) / pooling[1]
			print '\t\t\tOutput Dimension:    {}'.format( (kernels,input_dim[1],new_dim_1,new_dim_2) )

		self.input = layer_input
		self.output_dim = (kernels,input_dim[1],new_dim_1,new_dim_2)

		self.filter = ( kernels, kernel_stride, kernel_dim[0], kernel_dim[1] )
		
		# there are "num input feature maps * filter height * filter width" inputs to each hidden unit
		fan_in = np.prod( self.filter[1:] )
		# each unit in the lower layer receives a gradient from:
		# "num output feature maps * filter height * filter width" / pooling size
		# each output map is dim-by-dim / pool. and there are x kernels.
		fan_out = self.filter[0] * np.prod( self.filter[2:] ) / np.prod(pooling)


		# Weights - initialize weights with random weights
		W_bound = np.sqrt( 6. / (fan_in + fan_out) )
		if W_in is None:
			W_values = rng.uniform( low=-W_bound, high=W_bound, size=self.filter )  
		else:
			W_values = W_in
		
		self.W = theano.shared( value=np.asarray( W_values, dtype=theano.config.floatX ), borrow=True  )
		print '\t\t\tWeights:             {}'.format( self.W.shape.eval() ) # self.W.get_value() for values.


		# Biases
		if b_in is None:
			b_values = np.zeros( (self.filter[0],), dtype=theano.config.floatX )
		else:
			b_values = np.asarray( b_in, dtype=theano.config.floatX )
		
		self.b = theano.shared( value=b_values, borrow=True )
		print '\t\t\tBiases:              {}'.format( self.b.shape.eval() )


		# convolve input feature maps with filters
		# Docs: http://deeplearning.net/software/theano/library/tensor/nnet/conv.html#theano.tensor.nnet.conv.conv2d
		# <class 'theano.tensor.var.TensorVariable'>
		conv_out = theano.tensor.nnet.conv.conv2d( 
			input=layer_input, 
			filters=self.W, 
			filter_shape=self.filter, 
			image_shape=input_dim, 
			border_mode='valid' # what does this do exactly?
		)

		# downsample each feature map individually, using maxpooling
		# http://deeplearning.net/software/theano/library/tensor/signal/downsample.html
		# <class 'theano.tensor.var.TensorVariable'>
		pooled_out = theano.tensor.signal.downsample.max_pool_2d( input=conv_out, ds=pooling, ignore_border=True )

		# add the bias term. Since the bias is a vector (1D array), we first reshape it to a tensor of 
		# shape (1,n_filters,1,1). Each bias will thus be broadcasted across mini-batches and feature map width & height
		# self.output = theano.tensor.tanh( pooled_out + self.b.dimshuffle('x',0,'x','x') )
		# self.output = theano.tensor.nnet.softplus( pooled_out + self.b.dimshuffle('x',0,'x','x') )
		self.output = activation.fn( pooled_out + self.b.dimshuffle('x',0,'x','x') )

		# store parameters of this layer
		self.params = [ self.W, self.b ]
		##################

