import sys, os
# Adding reference to the scripts folder
currentFileDir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(currentFileDir + '/../../')

import utilities as utl
import numpy as np
from pybrain.tools.validation import Validator
from collections import Counter


if __name__ == "__main__":
	print ("Reading Testing Data for cancer problem")
	# Reading testing data
	testingData = utl.readDataSetAsMatrix(utl.BREAST_CANCER_TESTING_FILE, 1, ',')

	inputs = testingData[:, :9]
	outputs = testingData[:, 9:]

	neuralNetwork = utl.readModelFromLocation(\
		utl.BREAST_CANCER_MLP_MODEL_FILE
	)

	rows, numberOfFeatures = inputs.shape
	predictions = np.array([\
		np.round( neuralNetwork.activate( inputs[r] ) )
		for r in range(rows)
	])

	
	classificationRate =  Validator.classificationPerformance(\
		predictions,
		outputs
	)
	print("Evaluation for MLP Model (cancer problem)")
	print("Classification Rate = {}".format(classificationRate))

	mapper = lambda x: (int(x[0]), int(x[1]))
	confusionMatrix  = Counter([\
		str( mapper(c) )
		for c in zip(predictions, outputs)  
	])

	print("True Positives  = {}".format( confusionMatrix['(1, 1)'] ))
	print("False Positives = {}".format( confusionMatrix['(1, 0)'] ))
	print("False Negatives = {}".format( confusionMatrix['(0, 1)'] ))
	print("True Negatives = {}".format( confusionMatrix['(0, 0)'] ))