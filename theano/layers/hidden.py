import numpy as np
import theano

from activation import *
from weights import create_weights_uniform




class HiddenLayer(object):

	def __init__( self, rng, layer_input, n_in, n_out, activation=Sigmoid, W_in=None, b_in=None, use_bias=True, verbose=True ):
		
		if verbose:
			print '\t\t    --- Initialising HIDDEN Layer'
			print '\t\t\tInput size:          {}'.format( n_in )
			print '\t\t\tOutput size:         {}'.format( n_out )
			print '\t\t\tActivation function: {}'.format( activation.name )

		self.input = layer_input
		# self.activation = activation

		# Weights
		if W_in is None:
			W_val = create_weights_uniform( rng=rng, n_in=n_in, n_out=n_out, activation=activation )
			W_in = theano.shared( value=W_val, name='W', borrow=True )
			
		self.W = W_in

		# Biases
		if b_in is None:
			b_val = np.zeros( (n_out,), dtype=theano.config.floatX )
			b_in = theano.shared( value=b_val, name='b', borrow=True )

		self.b = b_in

		# Create the forward propogated zeta.	
		if use_bias:
			zeta = theano.tensor.dot(layer_input, self.W) + self.b
		else:
			zeta = theano.tensor.dot(layer_input, self.W)

		# Calculate the activation
		self.output = ( zeta if activation is None else activation.fn(zeta) )

		# Set the parameters of this layer.
		if use_bias:
			self.params = [ self.W, self.b ]
		else:
			self.params = [ self.W ]
		##################
	##################



def dropout_mask( rng, p, values ):
	"""
	p is the probablity of dropping a unit
	"""
	srng = theano.tensor.shared_randomstreams.RandomStreams( rng.randint(999999) )
	# p=1-p because 1's indicate keep and p is prob of dropping
	mask = srng.binomial( n=1, p=(1-p), size=values.shape )
	# The cast is important because int * float32 = float64 which pulls things off the gpu
	return values * theano.tensor.cast( mask, theano.config.floatX )
	##################



class DropoutHiddenLayer(HiddenLayer):
	
	def __init__( self, rng, layer_input, n_in, n_out, activation=Sigmoid, W_in=None, b_in=None, use_bias=True, dropout_rate=0.0 ):
		
		super(DropoutHiddenLayer, self).__init__( 
			rng=rng, 
			layer_input=layer_input, 
			n_in=n_in, 
			n_out=n_out,
			activation=activation,
			W_in=W_in, 
			b_in=b_in,
			use_bias=use_bias,
			verbose=False
		)

		self.output = dropout_mask( rng=rng, p=dropout_rate, values=self.output )
		##################
	##################


