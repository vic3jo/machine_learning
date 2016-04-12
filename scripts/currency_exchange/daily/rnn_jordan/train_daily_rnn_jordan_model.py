"""
Authors: Swati Bhartiya, Victor Trejo, and Utkarsh Bali
Description: Script to train the mlp model for the daily case.
"""

import sys, os
# Adding reference to the scripts folder
currentFileDir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(currentFileDir + '/../../../')
import utilities as utl
from pybrain.structure import LinearLayer

def trainModel(unitsInHiddenLayer = 8, debug = True):
	if debug:
		print("Reading Training Data RNN Jordan Model (Currency Exchange problem)")
	# Reading Training data
	trainData = utl.readDataSetAsMatrix(\
		utl.CURRENCY_EXCHANGE_TRAINING_FILE(\
			utl.SAMPLING_TYPE.AT_CLOSING_DAY
		)
	)

	inputs = trainData[:-1, 2:]

	if debug:
		print("Training Model RNN Jordan Model (Currency Exchange problem)")

	configuration = utl.RecurrentTrainProcessConfiguration()
	configuration.unitsInHiddenLayer = unitsInHiddenLayer
	configuration.maxEpochs = 200

	configuration.learningrate = 0.0001
	
	return utl.trainJordanRecurrentNetwork(\
		inputs, 1, configuration
	)


if __name__ == "__main__":

	model, errorsByEpoch = trainModel(4)

	print("Saving RNN Jordan Trained Model (Currency Exchange problem)")
	utl.saveModelAtLocation(
		model,
		utl.CURRENCY_EXCHANGE_RNN_JORDAN_MODEL_FILE(\
			utl.SAMPLING_TYPE.AT_CLOSING_DAY
		)
	)
