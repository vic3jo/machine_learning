import sys, os
# Adding reference to the scripts folder
currentFileDir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(currentFileDir + '/../')
sys.path.append(currentFileDir + '/mlp/')
sys.path.append(currentFileDir + '/rbf/')

import utilities as utl
import train_mlp_model, test_mlp_model 

class AverageClassifierStatistics(object):
	"""Average Classifier Statistics"""
	def __init__(\
		self
	):
		super(AverageClassifierStatistics, self).__init__()
		self.trainingTime = 0.0
		self.testingTime = 0.0
		self.classificationRate = 0.0
		self.trainingMemory = 0.0
		self.testingMemory = 0.0
		

base2Power = lambda r: map(lambda x: 2**x, r)

statistics = {\
	n:AverageClassifierStatistics()
	for n in base2Power(range(1, 6))
}

numberOfTries = 1

for unitsInHiddenLayer in base2Power(range(1, 6)):
	print "\n\nNumber Of Units in Hidden Layer  = {}".format(\
		unitsInHiddenLayer
	)

	trainingTimes, testingTimes = [], []
	classificationRates = []
	for i in range(numberOfTries):
		model, trainingMLPModelTime = utl.measureRunningTime(\
			lambda : train_mlp_model.trainModel(unitsInHiddenLayer, False)
		)

		classificationRate, testingMLPModelTime = utl.measureRunningTime(\
			lambda : test_mlp_model.evaluate(debug = False)
		)

		trainingTimes.append(trainingMLPModelTime)
		testingTimes.append(testingMLPModelTime)
		classificationRates.append(classificationRate)

	statistics[unitsInHiddenLayer].trainingTime = sum(trainingTimes)/len(trainingTimes)
	statistics[unitsInHiddenLayer].trainingTime = sum(testingTimes)/len(testingTimes)
	
	print "Average time taken training  = {} seconds".format()
	print "Average time taken testing  = {} seconds".format(sum(testingTimes)/len(testingTimes))
	print "Average classification rate {}".format(sum(classificationRates)/len(classificationRates))

# Evaluating memory
from memory_profiler import memory_usage
for numberOfUnitsInHiddenLayer in base2Power(range(1, 6)):
		memoryUsages = memory_usage(lambda : train_mlp_model.trainModel(numberOfUnitsInHiddenLayer, False))
		print "Memory Usage = {} MB".format(max(memoryUsages))