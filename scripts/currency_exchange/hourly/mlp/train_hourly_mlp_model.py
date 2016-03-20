import sys, os
# Adding reference to the scripts folder
currentFileDir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(currentFileDir + '/../../../')
import utilities as utl
from pybrain.structure import LinearLayer


def trainModel(unitsInHiddenLayer = 8):
	print("Reading Training Data MLP Model (Currency Exchange problem)")
	# Reading Training data
	trainData = utl.readDataSetAsMatrix(\
		utl.CURRENCY_EXCHANGE_TRAINING_FILE(\
			utl.SAMPLING_TYPE.HOURLY
		)
	)

	width = utl.HOURLY_WIDTH
	inputs = utl.createPattern(trainData[:-1, 2:], width)
	outputs = trainData[(width+1):, 2:]
	
	print("Training Model MLP Model (Currency Exchange problem)")
	configuration = utl.MLPTrainProcessConfiguration()
	configuration.unitsInHiddenLayer = unitsInHiddenLayer
	configuration.outputLayer = LinearLayer
	neuralNetwork = utl.trainMLPNetwork(\
		inputs,
		outputs,
		configuration
	)

	return neuralNetwork


if __name__ == "__main__":

	model = trainModel(8)

	print("Saving MLP Trained Model (Currency Exchange problem)")
	utl.saveModelAtLocation(
		model,
		utl.CURRENCY_EXCHANGE_MLP_MODEL_FILE(\
			utl.SAMPLING_TYPE.HOURLY
		)
	)
