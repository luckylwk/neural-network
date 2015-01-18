import numpy as np
import theano

from activation import SoftMax



class LogisticRegression(object):
	
	def __init__( self, layer_input, n_in, n_out, W_in=None, b_in=None, activation=SoftMax, verbose=True ):
		
		if verbose:
			print '\t\t    --- Initialising LOGISTIC REGRESSION Output Layer'
			print '\t\t\tInput size:          {}'.format( n_in )
			print '\t\t\tOutput size:         {}'.format( n_out )
			print '\t\t\tActivation function  {}'.format( activation.name )

		# initialize with 0 the weights W as a matrix of shape (n_in, n_out)
		if W_in is None:
			W_val = np.zeros( (n_in, n_out), dtype=theano.config.floatX )
			W_in = theano.shared( value=W_val, name='W', borrow=True )

		self.W = W_in
		
		# initialize the baises b as a vector of n_out 0s
		if b_in is None:
			b_val = np.zeros( (n_out,), dtype=theano.config.floatX )
			b_in = theano.shared( value=b_val, name='b', borrow=True )
		
		self.b = b_in

		# compute vector of class-membership probabilities in symbolic form
		self.p_y_given_x = activation.fn( theano.tensor.dot(layer_input, self.W) + self.b )

		# compute prediction as class whose probability is maximal in symbolic form
		self.y_pred = theano.tensor.argmax( self.p_y_given_x, axis=1 )

		# parameters of the model
		self.params = [ self.W, self.b ]
		##################


	def negative_log_likelihood(self, y):
		"""
		Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution. Mathematics::
            
            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D})
        
        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the correct label
        Note: we use the mean instead of the sum so that the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
		return -theano.tensor.mean( theano.tensor.log(self.p_y_given_x)[ theano.tensor.arange(y.shape[0]), y ] )
		##################


	def errors( self, y ):
		# Check for dimensions.
		if y.ndim != self.y_pred.ndim:
			raise TypeError( 'y should have the same shape as self.y_pred', ('y', target.type, 'y_pred', self.y_pred.type) )
		# check if y is of the correct datatype
		if y.dtype.startswith('int'):
			# the T.neq operator returns a vector of 0s and 1s, where 1 represents a mistake in prediction
			return theano.tensor.mean( theano.tensor.neq( self.y_pred, y ) )
		else:
			raise NotImplementedError()
		##################