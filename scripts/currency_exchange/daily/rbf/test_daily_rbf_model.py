import sys, os
from pybrain.tools.validation import Validator
import numpy as np
# Adding reference to the scripts folder
currentFileDir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(currentFileDir + '/../../../')
import utilities as utl

if __name__ == "__main__":
	print("Reading Training Data RBF Model (Currency exchange problem)")
	# Reading testing data
	testingData = utl.readDataSetAsMatrix(\
		utl.CURRENCY_EXCHANGE_TESTING_FILE(\
			utl.SAMPLING_TYPE.AT_CLOSING_DAY
		)
	)

	width = utl.DAILY_WIDTH
	inputs = utl.createPattern(testingData[:-1, 2:], width)
	# inputs = utl.normalize(inputs, 0, 1.0)
	outputs = testingData[(width+1):, 2:]
	neuralNetwork = utl.readModelFromLocation(\
		utl.CURRENCY_EXCHANGE_RBF_MODEL_FILE(\
			utl.SAMPLING_TYPE.AT_CLOSING_DAY
		)
	)

	print("Evaluation for RBF Model (currency exchange problem)")
	rows, numberOfFeatures = inputs.shape
	for r in xrange(rows):
		sample = inputs[r]
		predictedValue =  neuralNetwork.activate(sample)
		trueValue = outputs[r]
		print("True Value = {},  Predicted Value {}".format(\
			trueValue,
			predictedValue
		))


	predictions = np.array([\
		neuralNetwork.activate( inputs[r] ) 
		for r in range(rows)
	])


	utl.plotTimeSeries(outputs, predictions)
	print("MSE = {}".format(Validator.MSE(predictions, outputs)))