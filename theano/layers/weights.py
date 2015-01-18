import numpy as np
import theano




def create_weights( rng, n_in, n_out, activation=None ):
	# Bound ?
	W_bound = np.sqrt(6. / (n_in + n_out))
	# Create W_init from a Uniform distribution.
	W_init = np.asarray( rng.uniform( low=-W_bound, high=W_bound, size=(n_in, n_out) ), dtype=theano.config.floatX )
	# ??
	if activation is not None and activation.name == 'Sigmoid':
		W_init *= 4.0
	# Return the weights.
	return W_init
	##################