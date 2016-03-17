import sys, os

from pybrain.structure import LinearLayer
from pybrain.structure import GaussianLayer
# Adding reference to the scripts folder
currentFileDir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(currentFileDir + '/../')

import utilities as utl

def trainModel(unitsInHiddenLayer = 9):
	print ("Reading Training Data")
	# Reading Training data
	trainData = utl.readDataSetAsMatrix(utl.BREAST_CANCER_TRAINING_FILE, 1, ',')

	# Separating outputs from inputs
	inputs = utl.normalize(trainData[:, :9], 1, 10)
	outputs = trainData[:, 9:]

	# Remapping outputs
	outputs = (outputs - 2)/2
	
	print("Training RBF Model")
	return utl.trainNetwork(\
		inputs,
		outputs,
		unitsInHiddenLayer = unitsInHiddenLayer,
		builder = utl.RBF_CLASSIFIER_BUILDER,
		learningrate = 0.001,
		epochs = 100
	)



if __name__ == "__main__":

	model = trainModel(5)

	print ("Saving Trained Model")
	utl.saveModelAtLocation(\
		model,
		utl.BREAST_CANCER_RBF_MODEL_FILE
	)

