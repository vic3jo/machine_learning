"""
Authors: Swati Bhartiya, Victor Trejo, and Utkarsh Bali
Description: This file contains information used through out the project.
"""

import csv, os, pickle
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import SigmoidLayer
from pybrain.structure import LinearLayer
from pybrain.structure import GaussianLayer
from pybrain.structure import FeedForwardNetwork
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import FullConnection
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt

currentFileDir = os.path.dirname(os.path.abspath(__file__))



class SimpleMLPNetworkBuilder(object):
	"""SimpleMLPNetworkBuilder"""
	def __init__(self):
		super(SimpleMLPNetworkBuilder, self).__init__()
	
	def build(self,
		numberOfFeatures,
		outputSize,
		unitsInHiddenLayer = 2
	):
		return buildNetwork(\
			numberOfFeatures,
			unitsInHiddenLayer,
			outputSize,
			bias = True,
			outclass = SigmoidLayer,
			hiddenclass = SigmoidLayer
		)



class SimpleMLPWithLinearLayerNetworkBuilder(object):
	"""SimpleMLPWithLinearLayerNetworkBuilder"""
	def __init__(self):
		super(SimpleMLPWithLinearLayerNetworkBuilder, self).__init__()
	
	def build(self,
		numberOfFeatures,
		outputSize,
		unitsInHiddenLayer = 2
	):
		return buildNetwork(\
			numberOfFeatures,
			unitsInHiddenLayer,
			outputSize,
			bias = True,
			outclass = LinearLayer,
			hiddenclass = SigmoidLayer
		)


class RBFClassifierNetworkBuilder(object):
	"""RBFClassifierNetworkBuilder"""
	def __init__(self):
		super(RBFClassifierNetworkBuilder, self).__init__()
	

	def build(self,
		numberOfFeatures,
		outputSize,
		unitsInHiddenLayer = 2
	):
		neuralNetwork = FeedForwardNetwork()
		inputLayer = LinearLayer(numberOfFeatures)
		hiddenLayer = GaussianLayer(unitsInHiddenLayer)
		outputLayer = SigmoidLayer(outputSize)
		neuralNetwork.addInputModule(inputLayer)
		neuralNetwork.addModule(hiddenLayer)
		neuralNetwork.addOutputModule(outputLayer)

		inputToHiddenLayerConnection = FullConnection(\
			inputLayer,
			hiddenLayer
		)
		hiddenLayerToOuputConnection = FullConnection(\
			hiddenLayer,
			outputLayer
		)
		neuralNetwork.addConnection(inputToHiddenLayerConnection)
		neuralNetwork.addConnection(hiddenLayerToOuputConnection)
		neuralNetwork.sortModules()
		return neuralNetwork


MLP_BUILDER = SimpleMLPNetworkBuilder()
MLP_LINEAR_BUILDER =  SimpleMLPWithLinearLayerNetworkBuilder()
RBF_CLASSIFIER_BUILDER = RBFClassifierNetworkBuilder()


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

def normalize(values, minimum, maximum):
	middle = (maximum+minimum)/2
	differenceMiddle =  (maximum - minimum)/2
	return (values - middle)/differenceMiddle

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
	
	inputs = list(map(lambda x: tuple(x), inputs.tolist()))
	outputs = list(map(lambda x: tuple(x), outputs.tolist()))
	
	for r in range(rows):
		dataset.addSample(\
			inputs[r],
			outputs[r]
		)
	return dataset



def trainNetwork(\
	inputs,
	outputs,
	unitsInHiddenLayer = 2,
	builder = MLP_BUILDER,
	momentum = 0.1,
	epochs = 100,
	learningrate= 0.01
):
	rows, numberOfFeatures = inputs.shape
	rows, outputSize = outputs.shape

	neuralNetwork = builder.build(\
		numberOfFeatures,
		outputSize,
		unitsInHiddenLayer
	)
	dataset = __createSupervisedDataSet(\
		inputs,
		outputs
	)
	
	trainer = BackpropTrainer(\
		neuralNetwork,
		dataset,
		momentum = momentum,
		learningrate= 0.01
	)
	trainer.trainEpochs(epochs)
	return neuralNetwork

def readDataSetAsMatrix(\
	path,
	skipRows = 0,
	delimitier = ' '
):
	with open(path) as readFile: 
		return np.loadtxt(\
		readFile,
		skiprows = skipRows,
		delimiter = delimitier
	)


def saveModelAtLocation(model, location):
	with open(location, 'wb') as modelFile:
		pickle.dump(model, modelFile)


def readModelFromLocation(location):
	with open(location, 'rb') as modelFile:
		return pickle.load(modelFile)

def createPattern(data, width = 4):
	rows = data.ravel().shape[0]
	
	result = []
	for r in range(rows - width):
		result.append(data[r:r+width, 0])
	return np.array(result)

def plotTimeSeries(real, predicted, extraTitle=""):
	"""
	Plots the time series.
	Parameters:
			 real 
			 predicted 
	"""
	plt.subplot(211)
	plt.title(" real {}".format(extraTitle))
	plt.plot(real)
	plt.subplot(212)
	plt.title("predicted {}".format(extraTitle))
	plt.plot(predicted)
	plt.show()



# CURRENCY EXCHANGE CONSTANTS
SamplingType = namedtuple('SAMPLING_TYPE', ['AT_CLOSING_DAY', 'HOURLY'])
SAMPLING_TYPE = SamplingType(AT_CLOSING_DAY='at_closing_day', HOURLY='hourly')
CURRENCY_EXCHANGE_PROCESSED_DATA_FOLDER = '{}/../data/processed/currency_exchange/'.format(currentFileDir)

CURRENCY_EXCHANGE_TRAINING_FILE = lambda samplingType: '{}{}_training.dat'.format(CURRENCY_EXCHANGE_PROCESSED_DATA_FOLDER, samplingType)
CURRENCY_EXCHANGE_CROSS_VALIDATION_FILE = lambda samplingType: '{}{}_cross_validation.dat'.format(CURRENCY_EXCHANGE_PROCESSED_DATA_FOLDER, samplingType)
CURRENCY_EXCHANGE_TESTING_FILE = lambda samplingType: '{}{}_testing.dat'.format(CURRENCY_EXCHANGE_PROCESSED_DATA_FOLDER, samplingType)
CURRENCY_EXCHANGE_MODEL_FOLDER = '{}/../models/currency_exchange/'.format(currentFileDir)
CURRENCY_EXCHANGE_MLP_MODEL_FILE = lambda samplingType: '{}{}_mlp_model.pkl'.format(CURRENCY_EXCHANGE_MODEL_FOLDER, samplingType)
CURRENCY_EXCHANGE_RBF_MODEL_FILE = lambda samplingType: '{}{}_rbf_model.pkl'.format(CURRENCY_EXCHANGE_MODEL_FOLDER, samplingType)
DAILY_WIDTH = 5
HOURLY_WIDTH = 5

BREAST_CANCER_TRAINING_FILE = "{}/../data/processed/breast_cancer/training.csv".format(currentFileDir)
BREAST_CANCER_TESTING_FILE = "{}/../data/processed/breast_cancer/testing.csv".format(currentFileDir)
BREAST_CANCER_MLP_MODEL_FILE = "{}/../models/breast_cancer/breast_cancer_mlp_model.pkl".format(currentFileDir)
BREAST_CANCER_RBF_MODEL_FILE = "{}/../models/breast_cancer/breast_cancer_rbf_model.pkl".format(currentFileDir)








