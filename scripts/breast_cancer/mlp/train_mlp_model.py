import sys, os
# Adding reference to the scripts folder
currentFileDir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(currentFileDir + '/../../')

import utilities as utl

def trainModel(unitsInHiddenLayer = 8):
	print ("Reading Training Data for the cancer problem")
	# Reading Training data
	trainData = utl.readDataSetAsMatrix(utl.BREAST_CANCER_TRAINING_FILE, 1, ',')

	# Separating outputs from inputs
	inputs = trainData[:, :9]
	outputs = trainData[:, 9:]


	print("Training MLP Model for the cancer problem")
	return utl.trainMLPNetwork(\
		inputs,
		outputs,
		unitsInHiddenLayer = unitsInHiddenLayer
	)



if __name__ == "__main__":

	model = trainModel()

	print ("Saving MLP Trained Model for the cancer problem")
	utl.saveModelAtLocation(\
		model,
		utl.BREAST_CANCER_MLP_MODEL_FILE
	)
