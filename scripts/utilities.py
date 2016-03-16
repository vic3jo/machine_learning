"""
Authors: Swati Bhartiya, Victor Trejo, and Utkarsh Bali
Description: This file contains information used through out the project.
"""

import csv, os, pickle
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
import numpy as np
from collections import namedtuple

def readFileIgnoringLinesForCondition(fileLocation, shouldBeIgnored):
	'''
	Reads a file at a given location but ignores those line
	that satisfied certain condition.
	parameters:
		fileLocation -> location of the csv to read
		shouldBeIgnored -> a function that takes a string and returns a boolean
	returns: the read lines of the files except those that matched the 
			  given condition.
	'''
	validLines = []
	with open(fileLocation, 'rt') as sourceFile:
		for line in sourceFile:
			if not shouldBeIgnored(line):
				validLines.append(line.strip())
	return validLines

def readCSVFile(fileLocation):
	'''
	Reads a csv at a given location
	parameters:
		fileLocation -> location of the csv to read
	returns: the first row that corresponds to the header and the values rows.
	'''
	with open(fileLocation, 'rt') as sourceFile:
		csvReader = csv.reader(sourceFile)	
		rows = [row for row in csvReader]
		# Reading first row that corresponds to the header
		firstRow = rows.pop(0)
	return firstRow, rows


def writeCSVFile(rows, header, fileLocation):
	'''
	Writes a csv at a given location
	parameters:
		rows -> rows of the csv file
		header -> header of the csv file
		fileLocation -> location of the csv to writes
	'''
	with open(fileLocation, 'w') as destinationFile:
		writer = csv.writer(destinationFile)
		# Writting header
		writer.writerow(header)
		for row in rows: 
			writer.writerow(row)
		destinationFile.close()

def saveFileAtLocation(lines, fileLocation):
	with open(fileLocation, 'w') as destinationFile:
		for line  in lines:
			destinationFile.write(\
				"{}\n".format(line)
			)
	destinationFile.close()

def removeMissingValues(rows):
	'''
	This method is used to remove the rows
	 that have missing values
	parameters:
		rows -> rows to remove the missing values from.
	return the rows without those rows with missing values
	'''
	MISSING_VALUE_CHAR = '?'
	return [
		row for row in rows 
		if MISSING_VALUE_CHAR not in row
	]


def __createSupervisedDataSet(inputs, outputs):
	rows, numberOfFeatures = inputs.shape
	rows, outputSize = outputs.shape

	dataset = SupervisedDataSet(\
		numberOfFeatures,
		outputSize
	)
	
	inputs = map(lambda x: tuple(x), inputs.tolist())
	outputs = map(lambda x: tuple(x), outputs.tolist())
	
	for r in xrange(rows):
		dataset.addSample(\
			inputs[r],
			outputs[r]
		)
	return dataset


def trainNetwork(inputs, outputs, unitsInHiddenLayer = 2):
	rows, numberOfFeatures = inputs.shape
	rows, outputSize = outputs.shape

	neuralNetwork = buildNetwork(\
		numberOfFeatures,
		unitsInHiddenLayer,
		outputSize,
		bias = True
	)

	dataset = __createSupervisedDataSet(\
		inputs,
		outputs
	)
	
	trainer = BackpropTrainer(\
		neuralNetwork,
		dataset
	)
	trainer.trainUntilConvergence()

	return neuralNetwork

def readDataSetAsMatrix(path):
	with open(path) as readFile: return np.loadtxt(readFile)


def saveModelAtLocation(model, location):
	with open(location, 'w+') as modelFile:
		pickle.dump(model, modelFile)


def readModelFromLocation(location):
	with open(location) as modelFile:
		return pickle.load(modelFile)






# Some constants
currentFileDir = os.path.dirname(os.path.abspath(__file__))
SAMPLING_TYPE = {
	'At'
}

# CURRENCY EXCHANGE CONSTANTS
SamplingType = namedtuple('SAMPLING_TYPE', ['AT_CLOSING_DAY', 'HOURLY'])
SAMPLING_TYPE = SamplingType(AT_CLOSING_DAY='at_closing_day', HOURLY='hourly')
CURRENCY_EXCHANGE_PROCESSED_DATA_FOLDER = '{}/../data/processed/currency_exchange/'.format(currentFileDir)

CURRENCY_EXCHANGE_TRAINING_FILE = lambda samplingType: '{}{}_training.dat'.format(CURRENCY_EXCHANGE_PROCESSED_DATA_FOLDER, samplingType)
CURRENCY_EXCHANGE_CROSS_VALIDATION_FILE = lambda samplingType: '{}{}_cross_validation.dat'.format(CURRENCY_EXCHANGE_PROCESSED_DATA_FOLDER, samplingType)
CURRENCY_EXCHANGE_TESTING_FILE = lambda samplingType: '{}{}_testing.dat'.format(CURRENCY_EXCHANGE_PROCESSED_DATA_FOLDER, samplingType)

CURRENCY_EXCHANGE_MODEL_FOLDER = '{}/../models/currency_exchange/'.format(currentFileDir)
CURRENCY_EXCHANGE_MODEL_FILE = lambda samplingType: '{}{}_model.pkl'.format(CURRENCY_EXCHANGE_MODEL_FOLDER, samplingType)