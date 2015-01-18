# THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python run_mlp.py

import sys
from collections import OrderedDict


import numpy as np
import theano


sys.path.append('../') # theano folder path.

import datasets.mnist
from layers.activation import ReLU, SoftMax
from models.nn import NeuralNetwork
from trainers import GradientDescent

random_seed = 1234




if __name__ == '__main__':


	# LOAD the DATA.
	datasets = datasets.mnist.fn_T_load_data_MNIST( path_to_file='../../../_DATA/mnist.pkl.gz' )


	# BUILD the MODEL.
	rng = np.random.RandomState(random_seed)
	dropout = True
	x = theano.tensor.matrix('x') # <class 'theano.tensor.var.TensorVariable'>
	y = theano.tensor.ivector('y')

	# Create the MODEL.
	__MODEL__ = NeuralNetwork(
		rng=rng, 
		init_input=x,
		layer_sizes=[ 28*28, 1200, 1200, 10 ],
		dropout_rates=[ 0.2, 0.2, 0.2 ], # rate is the chance something is dropped.
		activations=[ ReLU, ReLU, SoftMax ],
		use_bias=True 
	)


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
		learning_rate_decay=learning_rate_decay,
		n_epochs=20
	)
	

