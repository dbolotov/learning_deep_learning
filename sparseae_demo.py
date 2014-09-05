from sparseae import *
import scipy.io
import numpy as np


def sample_images(numpatches=10,patchsize=8):
	"""
	sample random patches from .mat file
	"""
	img_dict = scipy.io.loadmat('data/IMAGES.mat')

	images = img_dict['IMAGES']

	imageside = images.shape[0]
	imagecount = images.shape[2]

	patches = np.zeros([patchsize ** 2, numpatches])

	for i in xrange(numpatches):
		x = np.random.randint(imageside-patchsize)
		y = np.random.randint(imageside-patchsize)
		z = np.random.randint(imagecount)

		sample = images[x:x+patchsize, y:y+patchsize,z]
		patches[:,i] = sample.flatten()

	return patches









# s = sparseae() #initialize with default settings
# s.initNParams()