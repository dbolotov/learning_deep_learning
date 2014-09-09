#!/usr/bin/python

"""
sparseae_functions.py

Functions to load, normalize and display data, and to check the gradient implementation
"""


from sparseae import *
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

def show_full_images(nrows=2,ncols=2):
	"""
	show full-size images
	"""
	img_dict = scipy.io.loadmat('data/IMAGES.mat')

	images = img_dict['IMAGES']

	imageside = images.shape[0]
	imagecount = images.shape[2]

	f,axes = plt.subplots(nrows=nrows,ncols=ncols)

	i = 0
	for axis in axes.flat:
		img = axis.imshow(images[:,:,i],cmap = plt.cm.gray, interpolation = 'nearest')

		axis.set_frame_on(False)
		axis.set_axis_off()
		i += 1

	plt.show()




def sample_images(numpatches=10000,patchsize=8):
	"""
	sample random patches from .mat file
	"""
	img_dict = scipy.io.loadmat('data/IMAGES.mat')

	images = img_dict['IMAGES']

	imageside = images.shape[0]
	imagecount = images.shape[2]

	patches = np.zeros([patchsize ** 2, numpatches])

	#random selection
	for i in xrange(numpatches):
		x = np.random.randint(imageside-patchsize)
		y = np.random.randint(imageside-patchsize)
		z = np.random.randint(imagecount)

		sample = images[x:x+patchsize, y:y+patchsize,z]
		patches[:,i] = sample.flatten()

	# #non-random selection
	# for i in xrange(numpatches):
	# 	x = i
	# 	y = i
	# 	z = i

	# 	sample = images[x:x+patchsize, y:y+patchsize,z]
	# 	patches[:,i] = sample.flatten(1)

	return patches

def normalize_data(patches):
	"""
	preprocessing for image patches
	"""

	patches = patches - np.mean(patches,0) #remove DC

	pstd = 3 * np.std(patches.flatten(),ddof=1)
	patches = np.maximum(np.minimum(patches,pstd), -pstd) * 1.0 / pstd #truncate to +/-3 standard deviations and scale to -1 to 1

	patches = (patches + 1) * 0.4 + 0.1; #rescale from [-1, 1] to [0.1, 0.9]

	return patches

def display_network(arr,patchsize=8,nrows=2,ncols=2):
	"""
	display some patches
	"""

	f,axes = plt.subplots(nrows=nrows,ncols=ncols)

	i = 0
	for axis in axes.flat:
		img = axis.imshow(arr[:,i].reshape(patchsize,patchsize),
			cmap = plt.cm.gray, interpolation = 'nearest')

		axis.set_frame_on(False)
		axis.set_axis_off()
		i += 1

	plt.show()


def display_weights(arr, nrows=5, ncols=5, vSide=8):
	"""
	display the trained weights
	"""
	figure, axes = plt.subplots(nrows = nrows, ncols = ncols)
	i = 0

	for axis in axes.flat:
		image = axis.imshow(arr[i, :].reshape(vSide, vSide),cmap = plt.cm.gray, interpolation = 'nearest')

		axis.set_frame_on(False)
		axis.set_axis_off()
		i += 1

	plt.show()

def computeNumericalGradient(J,theta):

	numgrad = np.zeros(theta.shape)

	eps = 1e-4
	n = numgrad.shape[0]
	i = np.eye(n,n)

	for k in xrange(numgrad.shape[0]):
		print numgrad.shape[0], k
		eps_vec = i[:,k] * eps

		val = (J(theta+eps_vec)[0] - J(theta-eps_vec)[0]) * 1.0 / (2*eps)

		numgrad[k] = val

	return numgrad

def simpleQuadraticFunction(x):

	value = x[0]**2 + 3*x[0]*x[1]
	grad = np.zeros((2,1))
	grad[0] = 2*x[0] + 3*x[1]
	grad[1] = 3*x[0]

	return value,grad

def checkNumericalGradient():
	x = np.array([[4],[10]])
	value,grad = simpleQuadraticFunction(x)

	numgrad = computeNumericalGradient(lambda x: simpleQuadraticFunction(x), x);

	print 'sizes: numgrad, grad:', numgrad.shape, grad.shape

	print np.hstack((numgrad,grad))

	diff = np.abs(numgrad-grad)/np.abs(numgrad+grad);
	print 'Norm of the difference between numerical and analytical gradient'
	print '(should be < 1e-9):'
	print diff


"""
Demo:
Uncomment the following 5 lines to instantiate an autoencoder and train it from this .py file
"""

# patches = sample_images()
# patches = normalize_data(patches)
# sae = sparseae() #initialize autoencoder with default settings
# weights = sae.train(patches) #train the network
# display_weights(arr = weights, vSide = sae.vSide, hSide = sae.hSide)


"""
Check gradient calculation:
Uncomment the following code to check the gradient implementation against a numerical computation.
"""
# patches = sample_images(numpatches=100,patchsize=8)
# patches = normalize_data(patches)
# sae = sparseae(hSide=2)
# [cost,grad] = sae.computeCost(sae.theta,patches)

# numgrad = computeNumericalGradient(lambda x: sae.computeCost(x,patches),sae.theta) # numerical gradient

# diff = np.abs(numgrad-grad) / np.abs(numgrad+grad)

# print 'these value should be very small (within a few magnitudes of 1e-9):'
# print 'max:',np.max(diff)
# print 'min:',np.min(diff)

# print 'the following two columns should be very similar'
# print np.transpose(np.vstack((numgrad,grad)))


