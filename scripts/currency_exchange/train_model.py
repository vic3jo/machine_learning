import sys, os
# Adding reference to the scripts folder
currentFileDir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(currentFileDir + '/../')
import utilities as utl


print "Reading Training Data"
# Reading Training data
trainingFilePath =  utl.CURRENCY_EXCHANGE_TRAINING_FILE(\
	utl.SAMPLING_TYPE.AT_CLOSING_DAY
)

trainData = utl.readDataSetAsMatrix(\
	trainingFilePath
)

trainData = trainData[:50, :]
inputs = trainData[:-1, 2:]
outputs = trainData[1:, 2:]
print "Training Model"
neuralNetwork = utl.trainNetwork(\
	inputs,
	outputs,
	unitsInHiddenLayer = 3
)

print "Saving Trained Model"
utl.saveModelAtLocation(
	neuralNetwork,
	utl.CURRENCY_EXCHANGE_MODEL_FILE(\
		utl.SAMPLING_TYPE.AT_CLOSING_DAY
	)
)
