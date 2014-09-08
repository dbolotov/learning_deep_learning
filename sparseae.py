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

		# random assignment
		self.W1 = np.random.rand(self.hSize, self.vSize) * 2 * self.r - self.r
		self.W2 = np.random.rand(self.vSize, self.hSize) * 2 * self.r - self.r

		#non-random assignment
		# self.W1 = np.ones((self.hSize, self.vSize)) * 2 * self.r - self.r
		# self.W2 = np.ones((self.vSize, self.hSize)) * 2 * self.r - self.r



		self.b1 = np.zeros([self.hSize,1])
		self.b2 = np.zeros([self.vSize,1])

		self.theta = np.hstack((self.W1.flatten(),self.W2.flatten(),self.b1.flatten(),self.b2.flatten()))


	def train(self,patches):
		"""
		train the ae
		"""
		cost,grad = self.computeCost(self.theta,patches)

	def computeCost(self,theta,data):
		"""
		compute cost of single pass
		"""
		W1 = theta[0:self.hSize*self.vSize].reshape(self.hSize,self.vSize)
		W2 = theta[self.hSize*self.vSize:2*self.hSize*self.vSize].reshape(self.vSize,self.hSize)
		b1 = theta[2*self.hSize*self.vSize:2*self.hSize*self.vSize+self.hSize].reshape(self.hSize,1)
		b2 = theta[2*self.hSize*self.vSize+self.hSize:].reshape(self.vSize,1)

		cost = 0

		W1grad = np.zeros(W1.shape)
		W2grad = np.zeros(W2.shape)
		b1grad = np.zeros(b1.shape)
		b2grad = np.zeros(b2.shape)

		m = data.shape[1]



		z_2 = np.dot(W1,data) + b1
		a_2 = self.__sigmoid(z_2)

		rho_hat = np.sum(a_2,1) * 1.0 / m 

		z_3 = np.dot(W2,a_2) + b2
		a_3 = self.__sigmoid(z_3)

		diff = a_3 - data

		sparse_penalty = self.__kl(self.sparsityParam, rho_hat)

		J_simple = sum(sum(diff ** 2))*1.0 / (2*m)

		reg = sum(W1.flatten() ** 2) + sum(W2.flatten() ** 2)*1.0


		cost = J_simple + self.beta * sparse_penalty + self.lam * reg/2

		delta_3 = diff * (a_3 * (1 - a_3))

		d2_simple = np.dot(np.transpose(W2),delta_3)
		
		d2_pen = self.__klDelta(self.sparsityParam,rho_hat)

		d2_pen = d2_pen.reshape(d2_pen.shape[0],1)

		delta_2 = (d2_simple + self.beta * d2_pen ) * a_2 * (1-a_2)


		b2grad = 1.0*np.sum(delta_3,1)/m
		b1grad = 1.0*np.sum(delta_2,1)/m


		W2grad = np.dot(delta_3,1.0*np.transpose(a_2)/m) + self.lam * W2

		W1grad = np.dot(delta_2,1.0*np.transpose(data)/m) + self.lam * W1


		grad = np.hstack((W1grad.flatten(),W2grad.flatten(),b1grad.flatten(),b2grad.flatten()))

		return cost,grad


	def __sigmoid(self,z):
		return 1.0 / (1.0 + np.exp(-z))

	def __kl(self,r,rh):

		return np.sum(r * np.log(r*1.0/rh) + (1-r) * np.log((1-r)*1.0/(1-rh)))

	def __klDelta(self,r,rh):
		return -1.0*r/rh + (1-r)*1.0/(1-rh)

