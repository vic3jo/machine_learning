import sys
import csv
import numpy as np
from random import shuffle

rows = []
indexList = []
'''
This method is used to read in the 
breast_cancer.csv file
'''
def readFile(file):
	f = open(file, 'rt')
	reader = csv.reader(f)	
	for r in reader:
		rows.append(r)
	firstRow = rows.pop(0)
	return firstRow
'''
This method is used to remove the rows
in the file which had missing values
'''
def removeMissingValues(firstRow):
	r = []
	for j in range(len(rows)):
		if '?' not in rows[j]:
			r.append(rows[j])	
	writeFile(r, firstRow)
'''
This method 
'''
def writeFile(r, firstRow):
	shuffle(r)
	train = ('training.csv')
	test = ('testing.csv')
	file = open(train, 'w', newline = '')
	writer = csv.writer(file)
	writer.writerow(firstRow)
	for i in range(len(r)):
		if i == 455:
			break
		writer.writerow(r[i])
	file = open(test, 'w', newline = '')
	writer = csv.writer(file)
	writer.writerow(firstRow)	
	while i < 683:
		writer.writerow(r[i])
		i = i + 1
		
def main():
	if len(sys.argv) != 2:
		print('Incorrect Format')
		return
	f = readFile(sys.argv[1])
	removeMissingValues(f)
main()