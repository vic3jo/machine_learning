"""
Authors: Swati Bhartiya, Victor Trejo, and Utkarsh Bali
Description: This file contains information used through out the project.
"""

import csv, os, pickle,  time, psutil
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer, RPropMinusTrainer
from pybrain.structure import SigmoidLayer
from pybrain.structure import LinearLayer
from pybrain.structure import GaussianLayer
from pybrain.structure import FeedForwardNetwork
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import FullConnection
from pybrain.auxiliary import kmeans
from pybrain.structure.modules.neuronlayer import NeuronLayer
from pybrain.tools.validation import Validator
from collections import Counter
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
from memory_profiler import memory_usage

currentFileDir = os.path.dirname(os.path.abspath(__file__))



class RBFTrainProcessConfiguration(object):
	"""docstring for RBFTrainProcessConfiguration"""
	def __init__(self):
		super(RBFTrainProcessConfiguration, self).__init__()
		self.unitsInHiddenLayer = 2
		self.maxEpochs = 100
		self.performClustering = True
		self.useClosestNeighbor = False
		self.outputLayer = SigmoidLayer
		self.variance = 1.0
		self.momentum = 0.9
		self.learningrate = 0.01
		self.trainer = RPropMinusTrainer


class MLPTrainProcessConfiguration(object):
	"""docstring for MLPTrainProcessConfiguration"""
	def __init__(self):
		super(MLPTrainProcessConfiguration, self).__init__()
		self.unitsInHiddenLayer = 2
		self.maxEpochs = 100
		self.outputLayer = SigmoidLayer
		self.momentum = 0.9
		self.learningrate = 0.01




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



def buildSimpleNetwork(
		numberOfFeatures,
		outputSize,
		outLayerClass
):
	neuralNetwork = FeedForwardNetwork()
	inputLayer = LinearLayer(numberOfFeatures)
	
	outputLayer = outLayerClass(outputSize)
	neuralNetwork.addInputModule(inputLayer)
	neuralNetwork.addOutputModule(outputLayer)

	inputLayerToOuputConnection = FullConnection(\
		inputLayer,
		outputLayer
	)

	neuralNetwork.addConnection(inputLayerToOuputConnection)
	neuralNetwork.sortModules()
	return neuralNetwork


class RBFNetwork(object):
	"""docstring for RBFNetwork"""
	def __init__(self, centers, variances, network):
		super(RBFNetwork, self).__init__()
		self.centers = centers
		self.variances = variances
		self.network = network

	def activate(self, inputValue):
		return self.network.activate(\
			allDistances(inputValue, self.centers, self.variances)
		)


def gaussian(x, xi, variance):
	v = (x - xi)
	vs = np.linalg.norm(x - xi)**2
	return np.exp(vs/variance)

def allDistances(x, centers, variances):
	rows, cols = centers.shape
	result = np.zeros(rows)
	for i in range(rows):
		result[i] = gaussian(x, centers[i], variances[i])
	return result





def trainRBFNetwork(\
	inputs,
	outputs,
	trainProcessConfiguration
):
	rows, numberOfFeatures = inputs.shape
	rows, outputSize = outputs.shape

	if trainProcessConfiguration.performClustering:
		
		centers, assignment =  kmeans.kmeanspp(\
			inputs,
			trainProcessConfiguration.unitsInHiddenLayer
		)

		variances = np.zeros(centers.shape[0])
		for i in range(centers.shape[0]):
			minimum = float("inf")
			distances = [\
				np.linalg.norm(centers[i]-centers[j])**2
				for j in range(centers.shape[0]) 
				if j != i
			]

			if trainProcessConfiguration.useClosestNeighbor:
				variances[i] = np.min(distances)
			else:	
				variances[i] = sum(distances)/len(distances)
	else:
		centers = inputs.copy()
		variances = np.ones(centers.shape[0]) * trainProcessConfiguration.variance
		trainProcessConfiguration.unitsInHiddenLayer = centers.shape[0]

	
	mappedInpus = np.apply_along_axis(\
		lambda x: allDistances(x, centers, variances),
		1,
		inputs
	)

	dataset = __createSupervisedDataSet(\
		mappedInpus,
		outputs
	)

	neuralNetwork = buildSimpleNetwork(\
		trainProcessConfiguration.unitsInHiddenLayer,
		outputSize, 
		trainProcessConfiguration.outputLayer
	)

	trainSet, crossValSet = dataset.splitWithProportion( 0.75 )
	
	trainer = trainProcessConfiguration.trainer(\
		neuralNetwork,
		dataset = trainSet
	)

	# trainer.trainUntilConvergence(\
	# 	maxEpochs = trainProcessConfiguration.maxEpochs
	# )

	network, errorsByEpoch = earlyStopTraining(
		crossValSet,
		neuralNetwork, 
		trainProcessConfiguration,
		trainer
	)
	return RBFNetwork(centers, variances, network), errorsByEpoch


def earlyStopTraining(crossValSet, neuralNetwork, configuration, trainer):
	# Training with early stop	
	previousStageError = float("inf")
	previousNetwork = neuralNetwork.copy()
	errorsByEpoch = []
	for epoch in range(1, configuration.maxEpochs + 1):
		predictions = neuralNetwork.activateOnDataset(crossValSet)
		# TODO: VERIFY
		if epoch % 5 == 0:
			validationError =  Validator.ESS(predictions, crossValSet['target'])
			if validationError > previousStageError: break
			previousStageError = validationError
			previousNetwork = neuralNetwork.copy()
		trainingError = trainer.train()
		errorsByEpoch.append(trainingError)
	return previousNetwork, errorsByEpoch

def trainMLPNetwork(\
	inputs,
	outputs,
	trainingConfiguration
):
	rows, numberOfFeatures = inputs.shape
	rows, outputSize = outputs.shape

	neuralNetwork = buildNetwork(\
			numberOfFeatures,
			trainingConfiguration.unitsInHiddenLayer,
			outputSize,
			bias = True,
			outclass = trainingConfiguration.outputLayer,
			hiddenclass = SigmoidLayer
	)

	dataset = __createSupervisedDataSet(\
		inputs,
		outputs
	)
	
	trainSet, crossValSet = dataset.splitWithProportion( 0.75 )

	trainer = BackpropTrainer(\
		neuralNetwork,
		trainSet,
		momentum = trainingConfiguration.momentum,
		learningrate = trainingConfiguration.learningrate
	)

	return earlyStopTraining(
		crossValSet,
		neuralNetwork, 
		trainingConfiguration,
		trainer
	)




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


def customPrint(content, turnedOn = True):
	if turnedOn:
		print(content)

def evaluateRegressionModel(model, inputs, outputs, label = '', debug = True):
	
	customPrint(\
		"Reading Training Data ({})".format(label),
		debug
	)

	customPrint(
		"Evaluation for ({})".format(label),
		debug
	)

	rows, numberOfFeatures = inputs.shape
	for r in xrange(rows):
		customPrint(\
			"True Value = {},  Predicted Value {}".format(\
				outputs[r],
				model.activate(inputs[r])
			),
			debug
		)


	predictions = np.array([\
		model.activate( inputs[r] ) 
		for r in range(rows)
	])

	if debug:
		plotTimeSeries(outputs, predictions)
	
	MSE = Validator.MSE(predictions, outputs)
	
	customPrint("MSE = {}".format(MSE), debug)
	return MSE

class ClassificationResult(object):
	"""ClassificationResult"""
	def __init__(self, predictions, outputs, classMapper):
		super(ClassificationResult, self).__init__()
		predictions = map(classMapper, predictions)
		outputs = map(classMapper, outputs)
		self.classes = set(predictions).union(set(outputs))
		matrix = {
			ct:{cp:0 for cp in self.classes}
			for ct in self.classes
		}
		for i in range(len(predictions)):
			ct = outputs[i]
			cp = predictions[i]
			matrix[ct][cp]+=1

		self.confusionMatrix = matrix
		self.classificationRate = Validator.classificationPerformance(\
			predictions,
			outputs
		)
	def __confusionMatrixRepresentation(self):
		header = ["{:^10}".format('T/P->')]
		for c in self.classes:
			header.append("{:^10}".format(c))
		header =  '|'.join(header)
		rows = []
		for ct in self.classes:
			content = ["{:^10} ".format(ct)]
			for cp in self.classes:
				content.append("{:^10}".format(self.confusionMatrix[ct][cp]))
			rows.append("|".join(content))
		body = '\n'.join(rows)
		return "{}\n{}".format(header, body)

	def __str__(self):
		return "\t\tClassification Rate = {} \n\n \t\tConfusion Matrix \n {}\n".format(
			self.classificationRate,
			self.__confusionMatrixRepresentation()
		)
		

def evaluateClassificationModel(model, inputs, outputs, label = '', debug = True, classMapper = lambda x: int(x[0])):
	
	customPrint(\
		"Reading Training Data ({})".format(label),
		debug
	)

	customPrint(
		"Evaluation for ({})".format(label),
		debug
	)

	rows, numberOfFeatures = inputs.shape

	predictions = np.array([\
		np.round( model.activate( inputs[r] ) )
		for r in range(rows)
	])

	

	result = ClassificationResult(predictions, outputs, classMapper)

	if debug:
		print(result)

	return result


def measureRunningTime(operation):
	"""
	Measures the time that takes an operation
	"""
	start = time.time()
	result = operation()
	end = time.time()
	timeTaken = end - start
	return result, timeTaken


def getMemoryUsage():
	"""
	Gets the memory usage of the script file.
	Returns:  the memory usage value in MB.
	"""
	currentProcess = psutil.Process(os.getpid())
	# To convert the data to MB
	MBNormalizer = float(2 ** 20)
	memoryInfo =  currentProcess.get_memory_info()[0]
	return memoryInfo/MBNormalizer



class AverageModelStatistics(object):
	"""Average Model Statistics"""
	def __init__(\
		self
	):
		super(AverageModelStatistics, self).__init__()
		self.trainingTime = 0.0
		self.testingTime = 0.0
		self.performance = None
		self.trainingMemory = 0.0
		self.testingMemory = 0.0
		self.trainingEpochs = 0
	
	def printValues(self):
		print("\t\tAverage epochs taken to train {}".format(self.trainingEpochs))
		print("\t\tAverage time taken training  = {} seconds".format(self.trainingTime))
		print("\t\tAverage time taken testing  = {} seconds".format(self.testingTime))
		print("\t\tAverage memory taken training  = {} MB".format(self.trainingMemory))
		print("\t\tAverage memory taken testing  = {} MB".format(self.testingMemory))
		print(self.performance)



def takeBestClassification(performances):
	best = None
	classificationRate = 0
	for p in performances:
		if p.classificationRate > classificationRate:
			best = p
			classificationRate = p.classificationRate
	return best



def evaluateNeuralNetworkForDifferentHiddenLayerSizes(\
	trainFunction,
	testFunction,
	hiddenLayerSizes,
	numberOfTries = 5,
	combinePerformance = takeBestClassification
):
	statistics = {\
		n:AverageModelStatistics()
		for n in hiddenLayerSizes
	}

	for unitsInHiddenLayer in hiddenLayerSizes:
		print "\tEvaluation for number Of Units in Hidden Layer = {}".format(\
			unitsInHiddenLayer
		)

		trainingTimes, testingTimes = [], []
		trainingMemoryUsages, testingMemoryUsages = [], []
		performances, trainingEpochs = [], []
		for i in range(numberOfTries):
			trainingResult, trainingMLPModelTime = measureRunningTime(\
				lambda : trainFunction(unitsInHiddenLayer, False)
			)

			model, errorsByEpoch = trainingResult

			performance, testingMLPModelTime = measureRunningTime(\
				lambda : testFunction(model, debug = False)
			)

			
			trainingMemoryUsages.append(\
				max(memory_usage(lambda : trainFunction(unitsInHiddenLayer, False)))
			) 

			testingMemoryUsages.append(
				max(memory_usage(lambda :  testFunction(model, debug = False)))
			)

			trainingTimes.append(trainingMLPModelTime)
			testingTimes.append(testingMLPModelTime)
			performances.append(performance)
			trainingEpochs.append(len(errorsByEpoch))

		statistics[unitsInHiddenLayer].trainingEpochs = sum(trainingEpochs)/len(trainingEpochs)
		statistics[unitsInHiddenLayer].trainingTime = sum(trainingTimes)/len(trainingTimes)
		statistics[unitsInHiddenLayer].testingTime = sum(testingTimes)/len(testingTimes)
		statistics[unitsInHiddenLayer].trainingMemory = sum(trainingMemoryUsages)/len(trainingMemoryUsages)
		statistics[unitsInHiddenLayer].testingMemory = sum(testingMemoryUsages)/len(testingMemoryUsages)
		statistics[unitsInHiddenLayer].performance = combinePerformance(performances)

	return statistics


def printStatistics(statistics):
	for unitsInHiddenLayer in sorted(statistics.keys()):
		print "\n\tNumber Of Units in Hidden Layer  = {}".format(\
			unitsInHiddenLayer
		)
		print "\t------------------------------------"
		statistics[unitsInHiddenLayer].printValues()




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








