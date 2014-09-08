from sparseae import *
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

def sample_images(numpatches=10,patchsize=8):
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

def display_network(arr,numtoshow=64,patchsize=8):
	"""
	display some patches
	"""

	numtoshow = np.minimum(64,numtoshow)
	plotsize = np.sqrt(numtoshow).astype(int)

	f,axes = plt.subplots(nrows=plotsize,ncols=plotsize)

	i = 0
	for axis in axes.flat:
		img = axis.imshow(arr[:,i].reshape(patchsize,patchsize),
			cmap = plt.cm.gray, interpolation = 'nearest')

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




patches = sample_images()
patches = normalize_data(patches)

s = sparseae() #initialize with default settings
s.initNParams()


print 'check cost calculation against numerical gradient'
[cost,grad] = s.computeCost(s.theta,patches)

numgrad = computeNumericalGradient(lambda x: s.computeCost(x,patches),s.theta)

diff = np.abs(numgrad-grad) / np.abs(numgrad+grad)

print 'these value should be less than 1e-9:'
print 'max:',np.max(diff)
print 'min:',np.min(diff)

# # print np.hstack((numgrad,grad))


