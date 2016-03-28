"""
Authors: Swati Bhartiya, Victor Trejo, and Utkarsh Bali
Description: Script to test the trained mlp  model for the cancer problem.
"""

import sys, os
# Adding reference to the scripts folder
currentFileDir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(currentFileDir + '/../../')
import utilities as utl

def evaluate(model, debug = True, inputs = None, outputs = None):
	if debug:
		print("Reading Testing Data for cancer problem")
	
	if inputs is None or outputs is None:
		# Reading testing data
		testingData = utl.readDataSetAsMatrix(utl.BREAST_CANCER_TESTING_FILE, 1, ',')

		inputs = testingData[:, :9]
		outputs = testingData[:, 9:]
	
	return utl.evaluateClassificationModel\
	(
		model,
		inputs,
		outputs,
		'MLP: cancer problem',
		debug
	)

if __name__ == "__main__":
	model = utl.readModelFromLocation\
	(
		utl.BREAST_CANCER_MLP_MODEL_FILE
	)
	evaluate(model)