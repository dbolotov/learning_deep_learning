"""
sparse autoencoder class

This is a python implementation of the autoencoder described here: 
http://ufldl.stanford.edu/wiki/index.php/Exercise:Sparse_Autoencoder
"""

import numpy as np

class sparseae(object):

	def __init__(self, visibleSize=8*8, hiddenSize = 25, 
					sparsityParam = 0.01, lam = 0.0001, beta = 3):
		"""
		initialize default ae params
		"""
		self.visibleSize = visibleSize		# number of input units 
		self.hiddenSize = hiddenSize		# number of hidden units 
		self.sparsityParam = sparsityParam	# desired average activation of the hidden units.
		self.lam = lam						# weight decay parameter       
		self.beta = beta					# weight of sparsity penalty term       


	def initNParams(self):
		"""
		initalize more params
		"""
		self.r = np.sqrt(6) / np.sqrt(self.hiddenSize + self.visibleSize + 1)

		self.W1 = np.random.rand(self.hiddenSize, self.visibleSize) * 2 * self.r - self.r
		self.W2 = np.random.rand(self.visibleSize, self.hiddenSize) * 2 * self.r - self.r

		self.b1 = np.zeros([self.hiddenSize,1])
		self.b2 = np.zeros([self.visibleSize,1])

		self.theta = np.hstack((self.W1.flatten(),self.W2.flatten(),self.b1.flatten(),self.b2.flatten()))

	def train(self):
		"""
		train the ae
		"""
		pass

	def __computeCost(self,patches):
		"""
		compute cost of single pass
		"""
		pass