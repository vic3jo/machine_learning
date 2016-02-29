"""
Authors: Swati Bhartiya, Victor Trejo, and Utkarsh Bali
Description: This file preprocessed data for the currency exchange problem.
"""

import sys, os, csv
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


def getClosingDaySamples(data):
	data = [tuple(r.split()) for r in data]
	samples = []
	for i in xrange(len(data)):
		if  i == (len(data)-1) or data[i][0]!=data[i+1][0]:
			samples.append(data[i])
	return samples

if __name__ == "__main__":
	firstSourceFileLocation = '{}/../../data/unprocessed/currency_exchange/C1-5.dat'.format(currentFileDir)
	secondSourceFileLocation = '{}/../../data/unprocessed/currency_exchange/C6-10.dat'.format(currentFileDir)

	trainingFileLocation = '{}/../../data/processed/currency_exchange/training.dat'.format(currentFileDir)
	testingFileLocation = '{}/../../data/processed/currency_exchange/testing.dat'.format(currentFileDir)
	
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

	sampledData = getClosingDaySamples(wholeData)

	values = [float(x[2]) for x in sampledData]
	total = len(values)
	trainingSize = (2 * total)/3
	trainingSet = values[:trainingSize]
	testingSet = values[trainingSize:]

	utl.saveFileAtLocation(\
		trainingSet,
		trainingFileLocation
	)

	utl.saveFileAtLocation(\
		testingSet,
		testingFileLocation
	)

	# plt.plot(trainingSet)
	# plt.ylabel('Training set')
	# plt.show()

	# print values
	# print len(wholeData)
