import sys


import numpy as np
import theano


sys.path.append('../') # theano folder path.

from layers.hidden import HiddenLayer
from layers.convolutional import ConvolutionalLayer
from layers.logisticregression import LogisticRegression



class ConvolutionalNeuralNetwork(object):
	
	def __init__( self, rng, init_input, input_dim, batch_size, layers, layer_sizes, use_bias=True ):
		
		print 100 * '-', '\n\t    --- Initialising MODEL: CONVOLUTIONAL NEURAL NETWORK'
		self.layers = []

		this_input = init_input.reshape( (batch_size, ) + input_dim )
		this_input_dim = ( batch_size, ) + input_dim
		

		for e,layer in enumerate(layers[:-1]):
			if layer['type'] == 'convolutional':
				conv_layer = ConvolutionalLayer( 
					rng=rng, 
					layer_input=this_input,
					input_dim=this_input_dim,
					kernels=layer['kernels'], 
					kernel_dim=layer['kernel_dim'],
					kernel_stride=layer['kernel_stride'],
					pooling=layer['pooling'],
					activation=layer['activation']
				)
				this_input = conv_layer.output
				this_input_dim = conv_layer.output_dim
				self.layers.append(conv_layer)
			else:
				this_input = this_input.flatten(2)
				hidden_layer = HiddenLayer(
					rng=rng,
					layer_input=this_input,
					n_in=layer_sizes[e], n_out=layer_sizes[e+1],
					activation=layer['activation']
				) # end of hiddenlayer
				this_input = hidden_layer.output
				self.layers.append(hidden_layer)


		# Create the OUTPUT layer.
		output_layer = LogisticRegression(
			rng=rng,
			layer_input=this_input, 
			n_in=layer_sizes[-1], n_out=layer_sizes[-1],
			activation=layers[-1]['activation'],
			verbose=True
		)
		self.layers.append(output_layer)

		
		# Set the COST and ERRORS for this model.
		self.negative_log_likelihood = output_layer.negative_log_likelihood
		self.errors = output_layer.errors


		# Grab all the parameters together.
		self.params = [ param for layer in self.layers for param in layer.params ]

		##################
	##################
