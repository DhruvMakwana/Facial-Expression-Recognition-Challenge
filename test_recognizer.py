#importing libraries to run this file execute following command
##  python test_recognizer.py --model checkpoints/vgg100/epoch_100.hdf5
from config import emotion_config as config
from preprocessing import imagetoarraypreprocessor
from hdf5 import hdf5datasetgenerator
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import argparse

#command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str,
	help="path to model checkpoint to load")
args = vars(ap.parse_args())

#initialize the testing data generator and image preprocessor
testAug = ImageDataGenerator(rescale=1 / 255.0)
iap = imagetoarraypreprocessor.ImageToArrayPreprocessor()

#initialize the testing dataset generator
testGen = hdf5datasetgenerator.HDF5DatasetGenerator(config.TEST_HDF5, 
	config.BATCH_SIZE,
	aug=testAug, 
	preprocessors=[iap], 
	classes=config.NUM_CLASSES)

#load the model from disk
print("[INFO] loading {}...".format(args["model"]))
model = load_model(args["model"])

#evaluate the network
(loss, acc) = model.evaluate_generator(
	testGen.generator(),
	steps=testGen.numImages // config.BATCH_SIZE,
	max_queue_size=config.BATCH_SIZE * 2)
print("[INFO] accuracy: {:.2f}".format(acc * 100))

#close the testing database
testGen.close()
