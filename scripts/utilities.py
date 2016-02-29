"""
Authors: Swati Bhartiya, Victor Trejo, and Utkarsh Bali
Description: This file contains information used through out the project.
"""

import csv

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

