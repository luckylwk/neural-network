import cPickle
import time
from math import sqrt

import numpy as np
import theano

from gpu import fn_theano_gpu_shared_dataset


def fn_theano_load_kaggle_national_science_bowl( path_to_file ):

	start_time = time.time()
	print 100 * '-', '\n\t    *** LOADING DATASET: KAGGLE National Science Bowl'

	try:
		f = open( path_to_file, 'rb')
		X, y_val, y_lbl  = cPickle.load(f)
		f.close()
	except:
		print '\t\tFailure: FILE DOES NOT EXIST', sys.exc_info()[0]

	# Change to correct data-types.
	X = np.asarray(X)
	m = X.shape[0]
	X = X.reshape( m, X.shape[2] )
	y_val = np.asarray(y_val).reshape( m, 1 )

	# Create a permutation and re-order the dataset.
	rng = np.random.RandomState(10)
	rnd = rng.permutation( m )
	X = X[rnd,:]
	y_val = y_val[rnd,:]

	# Create train/cv/test splits.
	m_train = int(0.7*m)
	m_cv = int( 0.5*(m-m_train) )
	m_test = m - m_train - m_cv
	train_set = ( X[:m_train,:], y_val[:m_train,:] )
	valid_set = ( X[m_train:(m_train+m_cv),:], y_val[m_train:(m_train+m_cv),:] )
	test_set = ( X[(m_train+m_cv):(m_train+m_cv+m_test),:], y_val[(m_train+m_cv):(m_train+m_cv+m_test),:] )

	# Create Theano shared variables (for GPU processing)
	test_set_x, test_set_y = fn_theano_gpu_shared_dataset(test_set)
	valid_set_x, valid_set_y = fn_theano_gpu_shared_dataset(valid_set)
	train_set_x, train_set_y = fn_theano_gpu_shared_dataset(train_set)

	# Print out information
	print '\t\tTime elapsed:      {:.2} seconds'.format( time.time()-start_time )
	print '\t\tTraining set:      {} {} {}'.format( train_set_x.get_value(borrow=True).shape, train_set[1].shape, type(train_set_x) )
	print '\t\tValidation set:    {} {}'.format( valid_set[0].shape, valid_set[1].shape )
	print '\t\tTesting set:       {} {}'.format( test_set[0].shape, test_set[1].shape )
	dim = int(sqrt(train_set_x.get_value(borrow=True).shape[1]))
	print '\t\tImage Dimensions:  {}-by-{}-by-1'.format(dim,dim)

	# Output is a list of tuples. Each tuple is filled with an m-by-n matrix and an m-by-1 array.
	return [ (train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y), (dim,dim,1) ]
	##################