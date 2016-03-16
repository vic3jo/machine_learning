import sys, os
# Adding reference to the scripts folder
currentFileDir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(currentFileDir + '/../')
import utilities as utl



print ("Reading Testing Data")
# Reading testing data


testingData = utl.readDataSetAsMatrix(\
	utl.Breast_cancer_testing_file,
	1,
	','
)


inputs = testingData[:, :9]
outputs = testingData[:, 9:]
outputs[outputs == 2] = 0
outputs[outputs == 4] = 1

neuralNetwork = utl.readModelFromLocation(utl.Breast_cancer_model_file)

rows, numberOfFeatures = inputs.shape
for r in range(rows):
	sample = inputs[r]
	predictedValue =  neuralNetwork.activate(sample)
	predictedValue = 1 if predictedValue >= 0.5 else 0
	trueValue = outputs[r]
	print ("True Value = {},  Predicted Value {}".format(\
		trueValue,
		predictedValue
	))
