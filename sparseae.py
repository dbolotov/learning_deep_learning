"""
sparse autoencoder class

This is a python implementation of the autoencoder described here: 
http://ufldl.stanford.edu/wiki/index.php/Exercise:Sparse_Autoencoder
"""

import numpy as np

class sparseae(object):

	def __init__(self, vSize=8*8, hSize = 25, 
					sparsityParam = 0.01, lam = 0.0001, beta = 3):
		"""
		initialize default ae params
		"""
		self.vSize = vSize		# number of input units 
		self.hSize = hSize		# number of hidden units 
		self.sparsityParam = sparsityParam	# desired average activation of the hidden units.
		self.lam = lam						# weight decay parameter       
		self.beta = beta					# weight of sparsity penalty term       


	def initNParams(self):
		"""
		initalize more params
		"""
		self.r = np.sqrt(6) / np.sqrt(self.hSize + self.vSize + 1)

		self.W1 = np.random.rand(self.hSize, self.vSize) * 2 * self.r - self.r
		self.W2 = np.random.rand(self.vSize, self.hSize) * 2 * self.r - self.r

		self.b1 = np.zeros([self.hSize,1])
		self.b2 = np.zeros([self.vSize,1])

		self.theta = np.hstack((self.W1.flatten(),self.W2.flatten(),self.b1.flatten(),self.b2.flatten()))

	def train(self):
		"""
		train the ae
		"""
		# self.computeCost(self.theta)

	def computeCost(self,theta,data):
		"""
		compute cost of single pass
		"""
		W1 = theta[0:self.hSize*self.vSize].reshape(self.hSize,self.vSize)
		W2 = theta[self.hSize*self.vSize:2*self.hSize*self.vSize].reshape(self.vSize,self.hSize)
		b1 = theta[2*self.hSize*self.vSize:2*self.hSize*self.vSize+self.hSize]
		b2 = theta[2*self.hSize*self.vSize+self.hSize:]

		# print W1.shape, W2.shape, b1.shape, b2.shape

		cost = 0

		W1grad = np.zeros(W1.shape)
		W2grad = np.zeros(W2.shape)
		b1grad = np.zeros(b1.shape)
		b2grad = np.zeros(b2.shape)



	def __sigmoid(self,z):
		return 1.0 / (1.0 + np.exp(-z))