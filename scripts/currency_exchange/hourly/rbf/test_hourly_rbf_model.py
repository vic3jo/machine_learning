import sys, os
import numpy as np
# Adding reference to the scripts folder
currentFileDir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(currentFileDir + '/../../../')
import utilities as utl

def evaluate(model, debug = True):
	# Reading testing data
	testingData = utl.readDataSetAsMatrix(\
		utl.CURRENCY_EXCHANGE_TESTING_FILE(\
			utl.SAMPLING_TYPE.HOURLY
		)
	)

	width = utl.HOURLY_WIDTH
	inputs = utl.createPattern(testingData[:-1, 2:], width)
	outputs = testingData[(width+1):, 2:]
	

	return utl.evaluateRegressionModel(\
		model,
		inputs,
		outputs,
		"RBF : Currency exchange problem 'hourly'",
		debug
	)

if __name__ == "__main__":
	model = utl.readModelFromLocation(\
		utl.CURRENCY_EXCHANGE_RBF_MODEL_FILE(\
			utl.SAMPLING_TYPE.HOURLY
		)
	)
	evaluate(model)