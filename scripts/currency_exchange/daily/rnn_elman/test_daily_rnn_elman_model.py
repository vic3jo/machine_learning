"""
Authors: Swati Bhartiya, Victor Trejo, and Utkarsh Bali
Description: Script to test the RNN Elman trained model for the daily case.
"""

import sys, os
import numpy as np
# Adding reference to the scripts folder
currentFileDir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(currentFileDir + '/../../../')
import utilities as utl

def evaluate(model, debug = True, inputs = None, outputs = None):
	
	if inputs is None or outputs is None:
		# Reading testing data
		testingData = utl.readDataSetAsMatrix(\
			utl.CURRENCY_EXCHANGE_TESTING_FILE(\
				utl.SAMPLING_TYPE.AT_CLOSING_DAY
			)
		)

		inputs = testingData[:-1, 2:]
		outputs = testingData[1:, 2:]

	

	return utl.evaluateRegressionModel(\
		model,
		inputs,
		outputs,
		'RNN Elman : Currency exchange problem',
		debug
	)


if __name__ == "__main__":
	model = utl.readModelFromLocation(\
		utl.CURRENCY_EXCHANGE_RNN_ELMAN_MODEL_FILE(\
			utl.SAMPLING_TYPE.AT_CLOSING_DAY
		)
	)
	evaluate(model)