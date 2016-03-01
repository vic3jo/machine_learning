"""
Authors: Swati Bhartiya, Victor Trejo, and Utkarsh Bali
Description: This file preprocessed data for the currency exchange problem.
			 The data is sampled using two sample rates daily at closing day and
			 hourly. Each sampling will yield three files for training, testing and cross validation.
			 These files are stored under data/processed/currency_exchange folder.
			 The data is normalized by dividing the rate output value by the max.
"""

import sys, os, csv, pywt
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle

# Adding reference to the scripts folder
currentFileDir = os.path.dirname(\
	os.path.abspath(__file__)
)
sys.path.append(\
	currentFileDir + '/../'
)

import utilities as utl


def denoise(signal, levels = 3):
	"""
	Applies Haar Wavelet transforms to denoise the data signal
	Parameters:
		signal -> signal to be denoised
		levels -> max level of decomposition.
	returns: the denoised signal.
	"""
	wavelet = pywt.Wavelet('haar')
	n = levels
	threshold = np.sqrt( 2*np.log( n*np.log2(n) ) )
	coefficients = pywt.wavedec(\
		signal,
		wavelet,
		level = n
	)
	# Soft Threshold function
	softThresholdFunction = lambda x: pywt.threshold(x, threshold)
	conservedCoefficients =  map(softThresholdFunction, coefficients)
	return pywt.waverec(conservedCoefficients, wavelet)

	

def getClosingDaySamples(data):
	"""
	Samples the data at sampling rate by date choosing the closing day value/
	parameters: 
				data -> data to sample

	return: The sampled data
	"""
	data = [tuple(r.split()) for r in data]
	samples = []
	for i in xrange(len(data)):
		if  i == (len(data)-1) or data[i][0]!=data[i+1][0]:
			samples.append(data[i])
	return samples


def getHourlyDataSamples(data):
	"""
	Samples the data at sampling rate by hour
	parameters: 
				data -> data to sample

	return: The sampled data
	"""
	data = [tuple(r.split()) for r in data]
	samples = []
	samplesForHour = []
	for i in xrange(len(data)):
		samplesForHour.append(data[i])
		if  i == (len(data)-1) or int(float(data[i][1]))!=int(float(data[i+1][1])):
			day = data[i][0]
			hour = str(int(float(data[i][1])))
			averageForHour = str(float(sum(float(x[2]) for x in samplesForHour))/len(samplesForHour))
			samples.append((day, hour, averageForHour))
			samplesForHour = []
	return samples


def plotTimeSeries(atClosingDaySampledData, hourlySampledData, extraTitle=""):
	"""
	Plots the time series for the sampling by day and hour.
	Parameters:
			 atClosingDaySampledData -> sampled data by day
			 hourlySampledData -> sampled data by hour
	"""
	hourlyValues =  [float(x[2]) for x in hourlySampledData]
	atClosingDayValues = [float(x[2]) for x in atClosingDaySampledData]
	plt.subplot(211)
	plt.title("Values Sampled at closing day {}".format(extraTitle))
	plt.plot(atClosingDayValues)
	plt.subplot(212)
	plt.title("Values Sampled hourly {}".format(extraTitle))
	plt.plot(hourlyValues)
	plt.show()

def processedFileNamesLocations(samplingType):
	"""
	Gets the location of the destination file for training, cross validation and
	testing, given a string that represents the type of sampling.
	Parameters:
				samplingType -> string that identifies  the type of sampling
	Returns: three string values: training file location, cross validation file location and
			 testing file location.
	"""
	dirName = '{}/../../data/processed/currency_exchange/'.format(currentFileDir)
	training = '{}{}_training.dat'.format(dirName, samplingType)
	crossValidation = '{}{}_cross_validation.dat'.format(dirName, samplingType)
	testing = '{}{}_testing.dat'.format(dirName, samplingType)
	return training, crossValidation, testing 


def storeProcessedData(data, samplingType):
	"""
	Stores the preprocessed data.
	It divides in to training(0.70 of data),
	cross validation (20\% of data) and 
	testing (10\% of the data), and stores
	it in three different files for each split.
	Parameters:
			data -> data to be stored
			samplingType -> to tag the files so it can 
							be differentiated from the other sampling 
							type files.
	"""
	total = len(data)
	# Splits the data into training, cross validation
	# and testing with the ratios 0.70, 0.20, 0.10 of the 
	# total 
	trainingSize = (7 * total)/10
	crossValidationSize = (2 * total)/10
	trainingSet = data[:trainingSize]
	crossValidationSet = data[trainingSize:(trainingSize+crossValidationSize)]
	testingSet = data[(trainingSize+crossValidationSize):]

	# Getting the names of the destination file.
	trainLoc, cVLoc, testLoc = processedFileNamesLocations(samplingType)
	trainingSet = [" ".join(x) for x in trainingSet]
	crossValidationSet = [" ".join(x) for x in crossValidationSet]
	testingSet = [" ".join(x) for x in testingSet]
	
	# Saving the split files.
	utl.saveFileAtLocation(trainingSet, trainLoc)
	utl.saveFileAtLocation(crossValidationSet, cVLoc)
	utl.saveFileAtLocation(testingSet, testLoc)

def normalize(data, xMax):
	"""
	Normalizes the exchange rate values by dividing each one by the max.
	Parameters:
		data -> data to be normalized
		xMax -> max value
	returns: the normalized data.
	"""
	days, hours, outputs = zip(*data)
	outputs = map(lambda x: str(float(x)/xMax), outputs)
	return zip(days, hours, outputs)


def denoiseOutputData(data, levels = 3):
	"""
	Denoises the output signal of the data.
	Parameters:
		data   -> data that contain the signal to denoise
		levels -> max level of decomposition of the signal
	Returns: the data with the signal denoised.
	"""
	days, hours, outputs = zip(*data) 
	outputs = denoise(outputs, levels)
	outputs = map(lambda x: str(x), outputs)
	return zip(days, hours, outputs)



if __name__ == "__main__":
	firstSourceFileLocation = '{}/../../data/unprocessed/currency_exchange/C1-5.dat'.format(currentFileDir)
	secondSourceFileLocation = '{}/../../data/unprocessed/currency_exchange/C6-10.dat'.format(currentFileDir)	
	ignoreLinesFunction = lambda l: 'set C part' in l

	dataFirstHalf = utl.readFileIgnoringLinesForCondition(\
			firstSourceFileLocation,
			ignoreLinesFunction
	)

	dataSecondHalf = utl.readFileIgnoringLinesForCondition(\
			secondSourceFileLocation,
			ignoreLinesFunction
	)

	wholeData = []
	wholeData.extend(dataFirstHalf)
	wholeData.extend(dataSecondHalf)
	# This value is going to be used as normalizer.
	maxExchangeRateValue = max(float(x.split()[2]) for x in wholeData)
	

	# Sampling at closing day and hourly.
	# For the hourly sampling an average for hour is calculated.
	atClosingDaySampledData = getClosingDaySamples(wholeData)
	hourlySampledData = getHourlyDataSamples(wholeData)
	
	# Plotting the time series after sampling
	plotTimeSeries(\
		atClosingDaySampledData,
		hourlySampledData
	)

	# Normalizing daily data
	atClosingDaySampledDataNormalized = normalize(\
		atClosingDaySampledData,
		maxExchangeRateValue
	)

	# Normalizing hourly data
	hourlySampledDataNormalized = normalize(\
		hourlySampledData,
		maxExchangeRateValue
	)

	# Denoising daily data
	atClosingDaySampledDataDenoised = denoiseOutputData(\
		atClosingDaySampledDataNormalized, 2
	)

	# Denoising hourly data
	hourlySampledDataDenoised = denoiseOutputData(\
		hourlySampledDataNormalized, 3
	)


	# Plotting the normalized and denoised time series after sampling
	plotTimeSeries(\
		atClosingDaySampledDataDenoised,
		hourlySampledDataDenoised,
		"Normalized and Denoised"
	)

	# Storing processed data
	storeProcessedData(atClosingDaySampledDataDenoised, 'at_closing_day')
	storeProcessedData(hourlySampledDataDenoised, 'hourly')
	
	


