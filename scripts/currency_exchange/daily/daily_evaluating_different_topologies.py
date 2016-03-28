"""
Authors: Swati Bhartiya, Victor Trejo, and Utkarsh Bali
Description: This is script to evaluate the different topologies
    of neural networks by changing the number of neurons in the hidden layer.
"""


import sys, os
# Adding reference to the scripts folder
currentFileDir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(currentFileDir + '/../../')
sys.path.append(currentFileDir + '/mlp/')
sys.path.append(currentFileDir + '/rbf/')

import utilities as utl
import train_daily_mlp_model, test_daily_mlp_model 
import train_daily_rbf_model, test_daily_rbf_model




base2Power = lambda r: map(lambda x: 2**x, r)
numberOfTries = 5
hiddenLayerSizes = base2Power(range(1, 6))

print("(Currency Exchange problem)")
print("\n\nEvaluating MLP:")
mlpStatistics = utl.evaluateNeuralNetworkForDifferentHiddenLayerSizes(\
	train_daily_mlp_model.trainModel,
	test_daily_mlp_model.evaluate,
	hiddenLayerSizes,
	numberOfTries,
	lambda x: "\t\tAverage MSE = {}".format(sum(x)/len(x))
)

hiddenLayerSizes.append(-1)
print("\n\nEvaluating RBF:")
rbfStatistics = utl.evaluateNeuralNetworkForDifferentHiddenLayerSizes(\
	train_daily_rbf_model.trainModel,
	test_daily_rbf_model.evaluate,
	hiddenLayerSizes,
	numberOfTries,
	lambda x: "\t\tAverage MSE = {}".format(sum(x)/len(x))
)

print("\n\nMLP statistics:")
utl.printStatistics(mlpStatistics)

print("\n\nRBF statistics:")
utl.printStatistics(rbfStatistics)