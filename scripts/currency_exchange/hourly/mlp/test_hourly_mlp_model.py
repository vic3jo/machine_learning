import sys, os
# Adding reference to the scripts folder
currentFileDir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(currentFileDir + '/../../../')
import utilities as utl

def evaluate(debug = True):
	# Reading testing data
	testingData = utl.readDataSetAsMatrix(\
		utl.CURRENCY_EXCHANGE_TESTING_FILE(\
			utl.SAMPLING_TYPE.HOURLY
		)
	)

	width = utl.HOURLY_WIDTH
	inputs = utl.createPattern(testingData[:-1, 2:], width)
	outputs = testingData[(width+1):, 2:]
	neuralNetwork = utl.readModelFromLocation(\
		utl.CURRENCY_EXCHANGE_MLP_MODEL_FILE(\
			utl.SAMPLING_TYPE.HOURLY
		)
	)

	return utl.evaluateRegressionModel(\
		neuralNetwork,
		inputs,
		outputs,
		"MLP : Currency exchange problem 'hourly'",
		debug
	)


if __name__ == "__main__":
	evaluate()