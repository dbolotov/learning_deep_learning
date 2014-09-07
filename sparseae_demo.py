from sparseae import *
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

def sample_images(numpatches=64,patchsize=8):
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



# s = sparseae() #initialize with default settings
# s.initNParams()

a = sample_images()

# display_network(a)

a = normalize_data(a)

# display_network(a)

s = sparseae() #initialize with default settings
s.initNParams()

s.train()