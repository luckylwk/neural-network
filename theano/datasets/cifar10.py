import os
import sys
import cPickle
import time
from math import sqrt

import numpy as np
import theano

from gpu import fn_theano_gpu_shared_dataset



def fn_theano_load_cifar10( path_to_dir ):
	'''
	Documentation: http://www.cs.toronto.edu/~kriz/cifar.html
	Dataset: http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

	Size: 60,000 color images.
	Image-dimensions: 32 x 32 x 3
	Classes: 10
	'''
	m = 60000
	m_train = 50000
	m_cv = 6000
	m_test = 4000
	img_shape = (3, 32, 32)

	start_time = time.time()
	print 100 * '-', '\n\t    *** LOADING DATASET: CIFAR10 Images'

	try:
		filenames = [ 'data_batch_1', 'data_batch_2', 'data_batch_3',
					  'data_batch_4', 'data_batch_5', 'test_batch']
		xs, ys = [], []
		for filename in filenames:
			with open( path_to_dir + filename, 'rb') as f:
				dic = cPickle.load(f)
				xs.append(dic['data'])
				ys.append(dic['labels'])
		# Create matrices.
		X = np.vstack(xs).reshape( (m,) + img_shape )
		y = np.hstack(ys).reshape(m,1)
		if X.shape[0] != y.shape[0] != m:
			raise RuntimeError('\t\tFailure: DATASET has incorrect dimensions.')
	except:
		print '\t\tFailure: FILE DOES NOT EXIST', sys.exc_info()[0]

	# Create a permutation and re-order the dataset.
	rng = np.random.RandomState(42)
	rnd = rng.permutation( m )
	X = X[rnd,:]
	y = y[rnd,:]

	train_set = ( X[:m_train,:], y[:m_train,:] )
	valid_set = ( X[m_train:(m_train+m_cv),:], y[m_train:(m_train+m_cv),:] )
	test_set = ( X[(m_train+m_cv):(m_train+m_cv+m_test),:], y[(m_train+m_cv):(m_train+m_cv+m_test),:] )

	# Create Theano shared variables (for GPU processing)
	test_set_x, test_set_y = fn_theano_gpu_shared_dataset(test_set)
	valid_set_x, valid_set_y = fn_theano_gpu_shared_dataset(valid_set)
	train_set_x, train_set_y = fn_theano_gpu_shared_dataset(train_set)

	# Print information.
	print '\t\tTime elapsed:      {:.2} seconds'.format( time.time()-start_time )
	print '\t\tTraining set:      {} {} | {}'.format( train_set_x.get_value(borrow=True).shape, train_set[1].shape, type(train_set_x) )
	print '\t\tValidation set:    {} {}'.format( valid_set[0].shape, valid_set[1].shape )
	print '\t\tTesting set:       {} {}'.format( test_set[0].shape, test_set[1].shape )
	print '\t\tImage Dimensions:  {}'.format( img_shape )

	# Output is a list of tuples. Each tuple is filled with an m-by-n matrix and an m-by-1 array.
	return [ (train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y), img_shape ]
	##################

