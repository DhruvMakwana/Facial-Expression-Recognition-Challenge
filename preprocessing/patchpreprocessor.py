#importing libraries
from sklearn.feature_extraction.image import extract_patches_2d

class PatchPreprocessor:
	def __init__(self, width, height):
		self.width = width
		self.height = height

	def preprocess(self, image):
		return extract_patches_2d(image, (self.height, self.width), 
			max_patches=1)[0]
	#Extracting a random patches of size self.width x self.height is 
	#easy using the extract_patches_2d function from the scikit-learn 
	#library
	