# Cost functions.
import numpy as np




class Cost_Quadratic:
	#
	name = 'Quadratic / MSE'
	#
	@staticmethod
	def fn( activations, Y ):
		return 0.5 * np.linalg.norm(activations-Y) ** 2
		##################
	@staticmethod
	def delta( activations, Y, activations_prime=None ):
		return (activations-Y) * activations_prime
		##################
	##################


class Cost_CrossEntropy:
	#
	name = 'Cross-Entropy / Logarithmic Loss'
	#
	@staticmethod
	def fn( activations, Y ):
		## LOG-LOSS | https://www.kaggle.com/wiki/LogarithmicLoss
		return np.nan_to_num( np.sum( -Y * np.log(activations) - (1.0-Y) * np.log(1.0-activations) ) )
		##################
	@staticmethod
	def delta( activations, Y, activations_prime=None ):
		return (activations-Y)
		##################
	##################