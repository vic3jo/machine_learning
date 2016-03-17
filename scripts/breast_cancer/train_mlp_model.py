import sys, os
# Adding reference to the scripts folder
currentFileDir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(currentFileDir + '/../')

import utilities as utl

def trainModel(unitsInHiddenLayer = 8):
	print ("Reading Training Data")
	# Reading Training data
	trainData = utl.readDataSetAsMatrix(utl.BREAST_CANCER_TRAINING_FILE, 1, ',')

	# Separating outputs from inputs
	inputs = utl.normalize(trainData[:, :9], 1, 10)
	outputs = trainData[:, 9:]

	# Remapping outputs
	outputs = (outputs - 2)/2

	print("Training Model")
	return utl.trainNetwork(\
		inputs,
		outputs,
		unitsInHiddenLayer = unitsInHiddenLayer
	)



if __name__ == "__main__":

	model = trainModel()

	print ("Saving MLP Trained Model")
	utl.saveModelAtLocation(\
		model,
		utl.BREAST_CANCER_MLP_MODEL_FILE
	)

