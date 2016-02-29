"""
Authors: Swati Bhartiya, Victor Trejo, and Utkarsh Bali
Description: This file preprocessed data for the breast cancer problem.
"""

import sys, os, csv
from random import shuffle

# Adding reference to the scripts folder
currentFileDir = os.path.dirname(\
	os.path.abspath(__file__)
)
sys.path.append(\
	currentFileDir + '/../'
)

import utilities as utl


if __name__ == "__main__":
	sourceFileLocation = '{}/../../data/unprocessed/breast_cancer/breast_cancer.csv'.format(currentFileDir)
	trainingDestinationFileLocation = '{}/../../data/processed/breast_cancer/training.csv'.format(currentFileDir)
	testingDestinationFileLocation = '{}/../../data/processed/breast_cancer/testing.csv'.format(currentFileDir)

	header, rows = utl.readCSVFile(sourceFileLocation)	
	rows = utl.removeMissingValues(rows)	

	# Removing sample id column from headers and value rows
	header = header[1:]
	rows = [row[1:] for row in rows]
	
	shuffle(rows)
	total = len(rows)

	# Two third of the total used for the training set
	trainingSize = (2 * total)/3 
	trainingSet = rows[:trainingSize]
	testingSet = rows[trainingSize:]

	# Saving training data set
	utl.writeCSVFile(\
		trainingSet,
		header,
		trainingDestinationFileLocation
	)
	
	# Saving testing data set
	utl.writeCSVFile(\
		testingSet,
		header,
		testingDestinationFileLocation
	)
	
