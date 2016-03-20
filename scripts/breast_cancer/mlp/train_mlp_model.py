import sys, os
# Adding reference to the scripts folder
currentFileDir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(currentFileDir + '/../../')

import utilities as utl

def trainModel(unitsInHiddenLayer = 8, debug = True):
	if debug:
		print ("Reading Training Data for the cancer problem")
	# Reading Training data
	trainData = utl.readDataSetAsMatrix(utl.BREAST_CANCER_TRAINING_FILE, 1, ',')

	# Separating outputs from inputs
	inputs = trainData[:, :9]
	outputs = trainData[:, 9:]

	if debug:
		print("Training MLP Model for the cancer problem")

	configuration = utl.MLPTrainProcessConfiguration()
	configuration.unitsInHiddenLayer = unitsInHiddenLayer
	return utl.trainMLPNetwork(\
		inputs,
		outputs,
		configuration
	)



if __name__ == "__main__":

	model, errorsByEpoch = trainModel()
	
	print ("Saving MLP Trained Model for the cancer problem")
	utl.saveModelAtLocation(\
		model,
		utl.BREAST_CANCER_MLP_MODEL_FILE
	)

