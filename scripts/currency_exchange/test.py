import sys, os
# Adding reference to the scripts folder
currentFileDir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(currentFileDir + '/../')
import utilities as utl


print "Reading Training Data"
# Reading testing data
testingFilePath =  utl.CURRENCY_EXCHANGE_TESTING_FILE(\
	utl.SAMPLING_TYPE.AT_CLOSING_DAY
)

testingData = utl.readDataSetAsMatrix(\
	testingFilePath
)


inputs = testingData[:-1, 2:]
outputs = testingData[1:, 2:]
neuralNetwork = utl.readModelFromLocation(\
	utl.CURRENCY_EXCHANGE_MODEL_FILE(\
		utl.SAMPLING_TYPE.AT_CLOSING_DAY
	)
)

rows, numberOfFeatures = inputs.shape
for r in xrange(rows):
	sample = inputs[r]
	predictedValue =  neuralNetwork.activate(sample)
	trueValue = outputs[r]
	print "True Value = {},  Predicted Value {}".format(\
		trueValue,
		predictedValue
	)
