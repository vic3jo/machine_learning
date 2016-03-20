import sys, os
# Adding reference to the scripts folder
currentFileDir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(currentFileDir + '/../../../')
import utilities as utl
import numpy as np
from pybrain.structure import LinearLayer

def trainModel():
	print("Reading Training Data RBF Model (Currency Exchange problem)")
	# Reading Training data
	trainData = utl.readDataSetAsMatrix(\
		utl.CURRENCY_EXCHANGE_TRAINING_FILE(\
			utl.SAMPLING_TYPE.HOURLY
		)
	)

	width = utl.HOURLY_WIDTH
	inputs = utl.createPattern(trainData[:-1, 2:], width)
	outputs = trainData[(width+1):, 2:].astype(np.float64)
	

	print("Training Model RBF Model (Currency Exchange problem)")
	configuration = utl.RBFTrainProcessConfiguration()
	configuration.outputLayer = LinearLayer
	configuration.performClustering = False
	configuration.maxEpochs = 100
	configuration.learningrate = 0.001

	return utl.trainRBFNetwork(\
		inputs,
		outputs,
		configuration
	)

	return neuralNetwork


if __name__ == "__main__":

	model = trainModel()

	print("Saving RBF Trained Model (Currency Exchange problem)")
	utl.saveModelAtLocation(
		model,
		utl.CURRENCY_EXCHANGE_RBF_MODEL_FILE(\
			utl.SAMPLING_TYPE.HOURLY
		)
	)
