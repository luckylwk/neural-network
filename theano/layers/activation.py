import numpy as np
import theano



# Classes for each activation function
class Sigmoid():
	#
	name = 'Sigmoid'
	#
	@staticmethod
	def fn( zeta ):
		return theano.tensor.nnet.sigmoid(x) # a = 1.0 / ( 1.0 + np.exp( -1.0*zeta ) )
		##################
	@classmethod
	def prime( cls, zeta ):
		return cls.fn( zeta=zeta ) * (1-cls.fn( zeta=zeta ))
		##################
	##################
	
	

class ReLU():
	#
	name = 'ReLU - Rectified Linear Unit'
	#
	@staticmethod
	def fn( zeta ):
		return theano.tensor.maximum( 0.0, zeta ) # np.amax([ np.ones(zeta.shape),zeta ],axis=0)
		##################
	@staticmethod
	def prime( zeta ):
		return 1.0 * ( zeta > 0 )
		##################
	##################



class Tanh():
	#
	name = 'Tanh'
	#
	@staticmethod
	def fn( zeta ):
		return theano.tensor.tanh(x) # np.tanh(zeta)
		##################
	@classmethod
	def prime( cls, zeta ):
		f = cls.fn( zeta=zeta )
		return 1.0 - np.multiply(f,f)
		##################
	##################



class SoftPlus():
	#
	name = 'SoftPlus'
	#
	@staticmethod
	def fn( zeta ):
		return theano.tensor.nnet.softplus( zeta )
		##################
	##################



class SoftMax():
	#
	name = 'SoftMax'
	#
	@staticmethod
	def fn( zeta ):
		# num = np.exp( zeta )
		# denom = num.sum(axis=1)
		# denom = denom.reshape( (zeta.shape[0],1) )
		# return num/denom
		return theano.tensor.nnet.softmax( zeta )
		##################
	@classmethod
	def prime( cls, zeta, vectorize=False ):
		return cls.fn( zeta=zeta ) * (1-cls.fn( zeta=zeta ))
		##################
	##################

