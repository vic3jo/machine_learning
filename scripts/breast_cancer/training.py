import sys, os
# Adding reference to the scripts folder
currentFileDir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(currentFileDir + '/../')
import utilities as utl
from pybrain.structure import SigmoidLayer

print ("Reading Training Data")
# Reading Training data
trainingFilePath =  utl.Breast_cancer_training_file


trainData = utl.readDataSetAsMatrix(\
	trainingFilePath,
	1,
	','
)

#trainData = trainData[:50, :]
inputs = trainData[:, :9]
outputs = trainData[:, 9:]
outputs[outputs == 2] = 0
outputs[outputs == 4] = 1

print("Training Model")

neuralNetwork = utl.trainNetwork(\
	inputs,
	outputs,
	unitsInHiddenLayer = 3,
	outputLayer = SigmoidLayer
)

print ("Saving Trained Model")
utl.saveModelAtLocation(
	neuralNetwork,
	utl.Breast_cancer_model_file
)
