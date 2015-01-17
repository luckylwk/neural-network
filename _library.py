import numpy as _np




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


class Act_Softmax:
	#
	name = 'Softmax'
	#
	@staticmethod
	def fn( zeta, vectorize=False ):
		num = _np.exp( zeta )
		denom = num.sum(axis=1)
		denom = denom.reshape( (zeta.shape[0],1) )
		return num/denom
		##################
	@classmethod
	def prime( cls, zeta, vectorize=False ):
		ap = cls.fn( zeta=zeta ) * (1-cls.fn( zeta=zeta ))
		return _np.vectorize(ap) if vectorize else ap
		##################
	##################



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
# Hinge loss. Useful for classification problems, maximizing the yes/no question. For example: Spam or no spam.