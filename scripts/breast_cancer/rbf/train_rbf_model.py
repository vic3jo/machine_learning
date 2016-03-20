import sys, os

from pybrain.structure import LinearLayer
from pybrain.structure import GaussianLayer
# Adding reference to the scripts folder
currentFileDir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(currentFileDir + '/../../')

import utilities as utl

def trainModel(unitsInHiddenLayer = 9):
	print ("Reading Training Data (Cancer Problem)")
	# Reading Training data
	trainData = utl.readDataSetAsMatrix(utl.BREAST_CANCER_TRAINING_FILE, 1, ',')

	# Separating outputs from inputs
	inputs = trainData[:, :9]
	outputs = trainData[:, 9:]
	
	print("Training RBF Model (Cancer problem)")
	configuration = utl.RBFTrainProcessConfiguration()
	configuration.unitsInHiddenLayer = unitsInHiddenLayer

	return utl.trainRBFNetwork(\
		inputs,
		outputs,
		configuration
	)



if __name__ == "__main__":

	model, errorsByEpoch = trainModel(2)
	
	print ("Saving RBF Trained Model (Cancer Problem)")
	utl.saveModelAtLocation(\
		model,
		utl.BREAST_CANCER_RBF_MODEL_FILE
	)

