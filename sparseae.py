"""
sparse autoencoder class
"""

class sparseae(self):

	def __init__(self):
		"""
		initialize default ae params
		"""
		self.visibleSize = 8*8;		# number of input units 
		self.hiddenSize = 25;		# number of hidden units 
		self.sparsityParam = 0.01;	# desired average activation of the hidden units.
		self.lam = 0.0001			# weight decay parameter       
		self.beta = 3				#weight of sparsity penalty term       
