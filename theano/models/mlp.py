import sys


import numpy as np
import theano


sys.path.append('../') # theano folder path.

from layers.hidden import HiddenLayer, DropoutHiddenLayer, dropout_mask
from layers.logisticregression import LogisticRegression



class MLP(object):
	
	def __init__( self, rng, init_input, layer_sizes, dropout_rates, activations, use_bias=True ):
		
		assert len(layer_sizes) - 1 == len(dropout_rates)

		print 100 * '-', '\n\t    --- Initialising MODEL: MULTI-LAYER-PERCEPTRON'

		self.weights = zip(layer_sizes[:-1], layer_sizes[1:])
		self.layers = []
		self.dropout_layers = []

		this_hidden_input = init_input
		this_dropout_input = dropout_mask( rng=rng, p=dropout_rates[0], values=init_input )

		# Create the HIDDEN LAYERS before the output layer.
		for e,(n_in,n_out) in enumerate(self.weights[:-1]):
			# Create the dropout layer.
			dropout_layer = DropoutHiddenLayer(
				rng=rng,
				layer_input=this_dropout_input,
				n_in=n_in, n_out=n_out, 
				activation=activations[e],
				W_in=None, b_in=None,
				use_bias=use_bias,
				dropout_rate=dropout_rates[e+1]
			) # end of dropout layer
			self.dropout_layers.append(dropout_layer)
			this_dropout_input = dropout_layer.output
			# Create a hidden layer.
			hidden_layer = HiddenLayer(
				rng=rng,
				layer_input=this_hidden_input,
				n_in=n_in, n_out=n_out,
				activation=activations[e],
				W_in=dropout_layer.W * (1. - dropout_rates[e+1]), # scale the weight matrix W with (1-p)
				b_in=dropout_layer.b,
				use_bias=use_bias 
			) # end of hiddenlayer
			# Append this layer to the list of layers
			self.layers.append(hidden_layer)
			# Set the output of this layer to be the input for the next layer.
			this_hidden_input = hidden_layer.output


		# Create the OUTPUT LAYER (Logistic Regression).
		dropout_output_layer = LogisticRegression(
				layer_input=this_dropout_input,
				n_in=self.weights[-1][0], n_out=self.weights[-1][1],
				W_in=None, b_in=None,
				activation=activations[-1]
		)
		self.dropout_layers.append(dropout_output_layer)
		normal_output_layer = LogisticRegression(
			layer_input=this_hidden_input, 
			n_in=self.weights[-1][0], n_out=self.weights[-1][1], 
			W_in=dropout_output_layer.W * (1. - dropout_rates[-1]), 
			b_in=dropout_output_layer.b,
			activation=activations[-1],
			verbose=False
		)
		self.layers.append(normal_output_layer)


		# Set the COST and ERRORS for this model.
		self.dropout_negative_log_likelihood = self.dropout_layers[-1].negative_log_likelihood
		self.dropout_errors = self.dropout_layers[-1].errors
		self.negative_log_likelihood = normal_output_layer.negative_log_likelihood
		self.errors = normal_output_layer.errors


		# Grab all the parameters together.
		# self.params = [ param for layer in self.layers for param in layer.params ]
		self.params = [ param for layer in self.dropout_layers for param in layer.params ]

		##################
	##################
