Project: Machine Learning and Neural Networks class group project
Authors: Swati Bhartiya, Victor Trejo, and Utkarsh Bali

Description: This project consists of solving multiple real world problems namely Cancer Prediction, 
Poker Hands classification and Exchange rate prediction, using Neural Networks.
			 			 
* Files:
    .
    :-- README : Current file. Contains basic information about the project.
    .
    :-- data  : this folder contains all the data files used in the project.
    .    :-- unprocessed : data before being processed
    .    .      .
    .    .      :-- breast_cancer: contains data sets for the breast cancer problem.
    .    .      :-- currency_exchange: contains data sets for the currency 
    .    .      .				exchange timeseries problem.
    .    .      :-- poker_hand: contains data sets for the poker hands problem.
    .    .
    .    :-- processed: data after being processed.
    .    .      :
    .    .      :-- breast_cancer: contains data sets for the breast 
    :    :      :    cancer problem after being processed.
    .    .      :-- currency_exchange: contains data sets for the currency 
    .    .      .				exchange timeseries problem after being processed.
    .    .      :-- poker_hand: contains data sets for the poker hands problem after being processed.
    .
    :-- scripts : This folder contains all scripts files used in the project.
    .    .
    .    :-- utilities.py :  this script contain utilities functions used through the project.
    .    .
    .	 :-- breast_cancer
    	 .    	.
    	 .		:- breast_cancer_preprocessing.py: script to preprocess breast cancer file.
         .      
         .
         :-- currency_exchange
    			.
    			:- currencty_preprocessing.py: script to preprocess currency exchange problem files.
    			

* How it works:
	In order to apply the preprocessing tasks to the raw data sets the following should be done:

	1. Run: python scripts/breast_cancer/breast_cancer_preprocessing.py
		This is going to create two files (training.csv, testing.csv) at data/processed/breast_cancer/ directory

	2. Run: python scripts/currency_exchange/currencty_preprocessing.py
		This is going to create 6 files:
			 at_closing_day_training.csv
			 at_closing_day_cross_validation.csv
			 at_closing_day_testing.csv
			 hourly_training.csv
			 hourly_cross_validation.csv
			 hourly_testing.csv

			 at data/processed/currency_exchange/ directory
             
    3. Run: python scripts preprocessingTesting.py
                AND
            python scripts preprocessingTraining.py



* Dependencies:
	1. Python 2.7    		  - https://www.python.org/downloads/    and    
								http://sourceforge.net/projects/numpy/files/NumPy/
	2. Python Numpy   		  -	http://www.numpy.org/
	3. Matplotlib			  - http://matplotlib.org/users/installing.html
	4. PyWavelets 			  - http://www.pybytes.com/pywavelets/dev/building_extension.html#

	