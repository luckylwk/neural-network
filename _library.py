import numpy as _np



# ## LOG-LOSS | https://www.kaggle.com/wiki/LogarithmicLoss
# def fn_logreg_cost( Theta, X, y, lmbda ):
# 	m = X.shape[0] # Number of training examples
# 	# Calculate the probability of each sample being 1.
# 	Prob = fn_sigmoid( X.dot(Theta) ) # <class 'numpy.ndarray'> - m-by-1.
# 	# Workaround for LOG(zero)
# 	PROB_1ST = _np.log(Prob)
# 	PROB_2ND = _np.log(1.0-Prob)
# 	PROB_1ST[PROB_1ST == -_np.inf] = 0
# 	PROB_2ND[PROB_2ND == -_np.inf] = 0
# 	# Calculate unregularized regression cost.
# 	J = ( PROB_1ST.dot(-y) - PROB_2ND.dot(1.0-y) ) / m
# 	# Calculate regularisation-penalty.
# 	J += lmbda * Theta.dot(Theta) / (2.0*m)
# 	return J
# 	##################


# ## Sigmoid function. zeta - <class 'numpy.ndarray'>
# def fn_sigmoid( zeta ):
# 	return 1.0 / ( 1.0 + _np.exp( -1.0*zeta ) )
# 	##################

# fn_sigmoid_vec = _np.vectorize(fn_sigmoid)
# ##################

# def fn_sigmoid_prime(z):
# 	return fn_sigmoid(z) * (1-fn_sigmoid(z))
# 	##################

# fn_sigmoid_prime_vec = _np.vectorize(fn_sigmoid_prime)
# ##################



class Act_Sigmoid:
	#
	name = 'Sigmoid'
	#
	@staticmethod
	def fn( zeta, vectorize=False ):
		a = 1.0 / ( 1.0 + _np.exp( -1.0*zeta ) )
		return _np.vectorize(a) if vectorize else a
		##################
	@classmethod
	def prime( cls, zeta, vectorize=False ):
		ap = cls.fn( zeta=zeta ) * (1-cls.fn( zeta=zeta ))
		return _np.vectorize(ap) if vectorize else ap
		##################
	##################


class Act_ReLU:
	#
	name = 'Rectified Linear Unit'
	#
	@staticmethod
	def fn( zeta, vectorize=False ):
		return _np.amax([ _np.ones(zeta.shape),zeta ],axis=0)
		##################
	@staticmethod
	def prime( zeta, vectorize=False ):
		return 1.0 * ( zeta > 0 )
		##################
	##################


class Act_Tanh:
	#
	name = 'Tanh'
	#
	@staticmethod
	def fn( zeta, vectorize=False ):
		return _np.tanh(zeta)
		##################
	@classmethod
	def prime( cls, zeta, vectorize=False ):
		f = cls.fn( zeta=zeta )
		return 1.0 - _np.multiply(f,f)
		##################
	##################


# class Act_Softplus:
# 	#
# 	name = 'Softplus'
# 	#
# 	@staticmethod
# 	def fn( zeta, vectorize=False ):
# 		return _np.log( 1.0 + _np.exp(zeta) )
# 		##################
# 	@classmethod
# 	def prime( cls, zeta, vectorize=False ):
# 		...
# 		##################
# 	##################


class Cost_Quadratic:
	#
	name = 'Quadratic'
	#
	@staticmethod
	def fn( activations, Y ):
		return 0.5 * _np.linalg.norm(activations-Y) ** 2
		##################
	@staticmethod
	def delta( activations, Y, activations_prime=None ):
		return (activations-Y) * activations_prime
		##################
	##################


class Cost_CrossEntropy:
	#
	name = 'Cross-Entropy'
	#
	@staticmethod
	def fn( activations, Y ):
		## LOG-LOSS | https://www.kaggle.com/wiki/LogarithmicLoss
		return _np.nan_to_num( _np.sum( -Y * _np.log(activations) - (1-Y) * _np.log(1-activations) ) )
		##################
	@staticmethod
	def delta( activations, Y, activations_prime=None ):
		return (activations-Y)
		##################
	##################





## QUANTILE-LOSS | http://en.wikipedia.org/wiki/Quantile_regression



# Squared loss. Useful for regression problems, when maximizing expectation. For example: Return on stocks.
# Classic loss. Vanilla squared loss (without the importance weight aware update).
# Hinge loss. Useful for classification problems, maximizing the yes/no question. For example: Spam or no spam.