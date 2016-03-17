import sys, os
# Adding reference to the scripts folder
currentFileDir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(currentFileDir + '/../')
import utilities as utl


def trainModel(unitsInHiddenLayer = 4):
	print("Reading Training Data")
	# Reading Training data
	trainData = utl.readDataSetAsMatrix(\
		utl.CURRENCY_EXCHANGE_TRAINING_FILE(\
			utl.SAMPLING_TYPE.AT_CLOSING_DAY
		)
	)

	inputs = trainData[:-1, 2:]
	outputs = trainData[1:, 2:]

	print("Training Model")
	neuralNetwork = utl.trainNetwork(\
		inputs,
		outputs,
		unitsInHiddenLayer = unitsInHiddenLayer,
		builder = utl.MLP_LINEAR_BUILDER,
		epochs = 1000
	)

	return neuralNetwork


if __name__ == "__main__":

	model = trainModel()

	print("Saving Trained Model")
	utl.saveModelAtLocation(
		model,
		utl.CURRENCY_EXCHANGE_MLP_MODEL_FILE(\
			utl.SAMPLING_TYPE.AT_CLOSING_DAY
		)
	)
