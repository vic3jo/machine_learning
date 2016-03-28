"""
Authors: Swati Bhartiya, Victor Trejo, and Utkarsh Bali
Description: script to evaluate the neural networks on noisy data.
"""

import sys, os
# Adding reference to the scripts folder
currentFileDir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(currentFileDir + '/../')
sys.path.append(currentFileDir + '/mlp/')
sys.path.append(currentFileDir + '/rbf/')

import utilities as utl
import train_mlp_model, test_mlp_model 
import train_rbf_model, test_rbf_model
from memory_profiler import memory_usage

def getTestData():
	testingData = utl.readDataSetAsMatrix(utl.BREAST_CANCER_TESTING_FILE, 1, ',')
	inputs = testingData[:, :9]
	outputs = testingData[:, 9:]
	return inputs, outputs



numberOfTries = 10
#  Percentages of noisy data.
percentageOfNoisyData = [0, 1, 5, 10]

print("Cancer Problem")
print("\n\nEvaluating MLP:")
mlpModel, epochs = train_mlp_model.trainModel(8)
mlpStatistics = utl.evaluateNetworkOnNoisyData(\
	mlpModel,
	test_mlp_model.evaluate,
	getTestData,
	percentageOfNoisyData,
	numberOfTries
)

print("\n\nEvaluating RBF:")
rbfModel, epochs = train_rbf_model.trainModel(8)
rbfStatistics = utl.evaluateNetworkOnNoisyData(\
	rbfModel,
	test_rbf_model.evaluate,
	getTestData,
	percentageOfNoisyData,
	numberOfTries
)

print("\n\nMLP statistics:")
utl.printStatistics(mlpStatistics, "\n\t Percentage of noisy data  = {}")

print("\n\nRBF statistics:")
utl.printStatistics(rbfStatistics, "\n\t Percentage of noisy data  = {}")