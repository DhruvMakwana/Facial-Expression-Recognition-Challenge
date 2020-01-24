#importing libraries
from keras.utils import np_utils
import numpy as np
import h5py

class HDF5DatasetGenerator:
	def __init__(self, dbPath, batchSize, preprocessors=None, 
		aug=None, binarize=True, classes=2):
		#store the batch size, preprocessors, and data augmentor,
		#whether or not the labels should be binarized, along with
		#the total number of classes
		#dbPath:The path to our HDF5 dataset that stores our images and 
		#corresponding class labels.

		#batchSize:The size of mini-batches to yield when training our 
		#network.

		#preprocessors: The list of image preprocessors we are going to apply 
		#(i.e., MeanPreprocessor, ImageToArrayPreprocessor, etc.).

		#aug:Defaulting to None, we could also supply a Keras 
		#ImageDataGenerator to apply data augmentation directly inside 
		#our HDF5DatasetGenerator.

		#binarize:Typically we will store class labels as single integers 
		#inside our HDF5 dataset; however, as we know, if we are applying 
		#categorical cross-entropy or binary cross-entropy as our loss 
		#function, we first need to binarize the labels as one-hot encoded 
		#vectors – this switch indicates whether or not this binarization 
		#needs to take place (which defaults to True).

		#classes:The number of unique class labels in our dataset. This value 
		#is required to accurately construct our one-hot encoded vectors 
		#during the binarization phase
		
		self.batchSize = batchSize
		self.preprocessors = preprocessors
		self.aug = aug
		self.binarize = binarize
		self.classes = classes

		self.db = h5py.File(dbPath)
		self.numImages = self.db["labels"].shape[0]

	def generator(self, passes=np.inf):
		#initialize the epoch count
		epochs = 0
		#keep looping infinitely --  the model will stop once we have
		#reach the desired number of epochs
		while epochs < passes:
			#loop over the HDF5 dataset
			for i in np.arange(0, self.numImages, self.batchSize):
				#extract the images and labels from the HDF dataset
				images = self.db["images"][i: i + self.batchSize]
				labels = self.db["labels"][i: i + self.batchSize]

				#check to see if the labels should be binarized
				if self.binarize:
					labels = np_utils.to_categorical(labels, self.classes)

				#check to see if our preprocessors are not None
				if self.preprocessors is not None:
					#initialize the list of processed images
					procImages = []

					#loop over the images
					for image in images:
						#loop over the preprocessors and apply each
						#to the image
						for p in self.preprocessors:
							image = p.preprocess(image)

						#update the list of processed images
						procImages.append(image)
					#update the images array to be the processed
					#images
					images = np.array(procImages)

				#if the data augmenator exists, apply it
				if self.aug is not None:
					(images, labels) = next(self.aug.flow(images,
						labels, batch_size=self.batchSize))
				#yield a tuple of images and labels
				yield (images, labels)

			#increment the total number of epochs
			epochs += 1
	def close(self):
		#close the database
		self.db.close()