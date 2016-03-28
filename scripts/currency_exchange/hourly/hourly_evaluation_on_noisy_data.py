"""
Authors: Swati Bhartiya, Victor Trejo, and Utkarsh Bali
Description: script to evaluate the neural networks on noisy data.
"""

import sys, os, random
# Adding reference to the scripts folder
currentFileDir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(currentFileDir + '/../../')
sys.path.append(currentFileDir + '/mlp/')
sys.path.append(currentFileDir + '/rbf/')


import utilities as utl
import train_hourly_mlp_model, test_hourly_mlp_model 
import train_hourly_rbf_model, test_hourly_rbf_model
from memory_profiler import memory_usage

def getTestData():
	testingData = utl.readDataSetAsMatrix(\
		utl.CURRENCY_EXCHANGE_TESTING_FILE(\
			utl.SAMPLING_TYPE.HOURLY
		)
	)

	width = utl.HOURLY_WIDTH
	inputs = utl.createPattern(testingData[:-1, 2:], width)
	outputs = testingData[(width+1):, 2:]

	return inputs, outputs



numberOfTries = 10
percentageOfNoisyData = [0, 1, 5, 10]

print("(Currency Exchange problem)")
print("\n\nEvaluating MLP:")
mlpModel, epochs = train_hourly_mlp_model.trainModel(4)
mlpStatistics = utl.evaluateNetworkOnNoisyData(\
	mlpModel,
	test_hourly_mlp_model.evaluate,
	getTestData,
	percentageOfNoisyData,
	numberOfTries,
	lambda x: "\t\tAverage MSE = {}".format(sum(x)/len(x))
)

print("\n\nEvaluating RBF:")
rbfModel, epochs = train_hourly_rbf_model.trainModel()
rbfStatistics = utl.evaluateNetworkOnNoisyData(\
	rbfModel,
	test_hourly_rbf_model.evaluate,
	getTestData,
	percentageOfNoisyData,
	numberOfTries,
	lambda x: "\t\tAverage MSE = {}".format(sum(x)/len(x))
)

print("\n\nMLP statistics:")
utl.printStatistics(mlpStatistics, "\n\t Percentage of noisy data  = {}")

print("\n\nRBF statistics:")
utl.printStatistics(rbfStatistics, "\n\t Percentage of noisy data  = {}")