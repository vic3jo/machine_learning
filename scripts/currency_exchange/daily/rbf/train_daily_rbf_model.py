import sys, os
# Adding reference to the scripts folder
currentFileDir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(currentFileDir + '/../../../')
import utilities as utl
import numpy as np
from pybrain.structure import LinearLayer

def trainModel(unitsInHiddenLayer = -1, debug = True):
	if debug:
		print("Reading Training Data RBF Model (Currency Exchange problem)")
	# Reading Training data
	trainData = utl.readDataSetAsMatrix(\
		utl.CURRENCY_EXCHANGE_TRAINING_FILE(\
			utl.SAMPLING_TYPE.AT_CLOSING_DAY
		)
	)

	width = utl.DAILY_WIDTH
	inputs = utl.createPattern(trainData[:-1, 2:], width)
	outputs = trainData[(width+1):, 2:].astype(np.float64)
	if debug:
		print("Training Model RBF Model (Currency Exchange problem)")
	configuration = utl.RBFTrainProcessConfiguration()
	configuration.outputLayer = LinearLayer
	if unitsInHiddenLayer <= 0:
		configuration.performClustering = False
	else:
		configuration.unitsInHiddenLayer = unitsInHiddenLayer
		configuration.performClustering = True
	configuration.maxEpochs = 1000 
	
	return utl.trainRBFNetwork(\
		inputs,
		outputs,
		configuration
	)


if __name__ == "__main__":

	model, errorsByEpoch = trainModel()

	print("Saving RBF Trained Model (Currency Exchange problem)")
	utl.saveModelAtLocation(
		model,
		utl.CURRENCY_EXCHANGE_RBF_MODEL_FILE(\
			utl.SAMPLING_TYPE.AT_CLOSING_DAY
		)
	)
