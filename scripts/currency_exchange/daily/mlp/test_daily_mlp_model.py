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

		width = utl.DAILY_WIDTH
		inputs = utl.createPattern(testingData[:-1, 2:], width)
		outputs = testingData[(width+1):, 2:]

	

	return utl.evaluateRegressionModel(\
		model,
		inputs,
		outputs,
		'MLP : Currency exchange problem',
		debug
	)


if __name__ == "__main__":
	model = utl.readModelFromLocation(\
		utl.CURRENCY_EXCHANGE_MLP_MODEL_FILE(\
			utl.SAMPLING_TYPE.AT_CLOSING_DAY
		)
	)
	evaluate(model)