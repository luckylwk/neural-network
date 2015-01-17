import numpy as np




class Act_Sigmoid:
	#
	name = 'Sigmoid'
	#
	@staticmethod
	def fn( zeta, vectorize=False ):
		a = 1.0 / ( 1.0 + np.exp( -1.0*zeta ) )
		return np.vectorize(a) if vectorize else a
		##################
	@classmethod
	def prime( cls, zeta, vectorize=False ):
		ap = cls.fn( zeta=zeta ) * (1-cls.fn( zeta=zeta ))
		return np.vectorize(ap) if vectorize else ap
		##################
	##################


class Act_ReLU:
	#
	name = 'Rectified Linear Unit'
	#
	@staticmethod
	def fn( zeta, vectorize=False ):
		return np.amax([ np.ones(zeta.shape),zeta ],axis=0)
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
		return np.tanh(zeta)
		##################
	@classmethod
	def prime( cls, zeta, vectorize=False ):
		f = cls.fn( zeta=zeta )
		return 1.0 - np.multiply(f,f)
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
		num = np.exp( zeta )
		denom = num.sum(axis=1)
		denom = denom.reshape( (zeta.shape[0],1) )
		return num/denom
		##################
	@classmethod
	def prime( cls, zeta, vectorize=False ):
		ap = cls.fn( zeta=zeta ) * (1-cls.fn( zeta=zeta ))
		return np.vectorize(ap) if vectorize else ap
		##################
	##################
