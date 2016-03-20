import sys, os
from pybrain.tools.validation import Validator
import numpy as np
# Adding reference to the scripts folder
currentFileDir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(currentFileDir + '/../../../')
import utilities as utl

if __name__ == "__main__":
	print("Reading Training Data MLP Model (Currency exchange problem)")
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

	print("Evaluation for MLP Model (currency exchange problem)")
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