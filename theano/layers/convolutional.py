class ConvLayer(object):
	
	def __init__( self, rng, input, filter_shape, image_shape, poolsize=(2, 2), W_in=None, b_in=None ):
		# input = 4D Tensor <class 'theano.tensor.var.TensorVariable'>
		# filter_shape = 4-var <type 'tuple'> - #kernels, layers?, kernel dim (2)
		# image_shape = 4-var <type 'tuple'>
		print '    --- Initialising CONVOLUTIONAL Layer'
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