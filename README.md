Project: Machine Learning and Neural Networks class group project
Authors: Swati Bhartiya, Victor Trejo, and Utkarsh Bali

Description: This project consists of solving multiple real world problems namely Cancer Prediction, 
Poker Hands classification and Exchange rate prediction, using Neural Networks.

Files:
.
├── data
│   ├── processed
│   │   ├── breast_cancer
│   │   │   ├── testing.csv
│   │   │   └── training.csv
│   │   ├── currency_exchange
│   │   │   ├── at_closing_day_testing.dat
│   │   │   ├── at_closing_day_training.dat
│   │   │   ├── hourly_testing.dat
│   │   │   └── hourly_training.dat
│   │   └── poker_hand
│   │       ├── poker-hand-testing.csv
│   │       └── poker-hand-training-true.csv
│   └── unprocessed
│       ├── breast_cancer
│       │   └── breast_cancer.csv
│       ├── currency_exchange
│       │   ├── C1-5.dat
│       │   └── C6-10.dat
│       └── poker_hand
│           ├── poker-hand-testing.data
│           └── poker-hand-training-true.data
├── models
│   ├── breast_cancer
│   │   ├── breast_cancer_mlp_model.pkl
│   │   └── breast_cancer_rbf_model.pkl
│   ├── currency_exchange
│   │   ├── at_closing_day_mlp_model.pkl
│   │   ├── at_closing_day_rbf_model.pkl
│   │   ├── hourly_mlp_model.pkl
│   │   └── hourly_rbf_model.pkl
│   └── poker_hand
|
├── README.md
└── scripts
    ├── breast_cancer
    │   ├── breast_cancer_preprocessing.py
    │   ├── evaluating_different_topologies.py
    │   ├── evaluation_on_noisy_data.py
    │   ├── mlp
    │   │   ├── test_mlp_model.py
    │   │   └── train_mlp_model.py
    │   └── rbf
    │       ├── test_rbf_model.py
    │       └── train_rbf_model.py
    ├── currency_exchange
    │   ├── currency_exchange_preprocessing.py
    │   ├── daily
    │   │   ├── daily_evaluating_different_topologies.py
    │   │   ├── daily_evaluation_on_noisy_data.py
    │   │   ├── mlp
    │   │   │   ├── test_daily_mlp_model.py
    │   │   │   └── train_daily_mlp_model.py
    │   │   └── rbf
    │   │       ├── test_daily_rbf_model.py
    │   │       └── train_daily_rbf_model.py
    │   └── hourly
    │       ├── hourly_evaluating_different_topologies.py
    │       ├── hourly_evaluation_on_noisy_data.py
    │       ├── mlp
    │       │   ├── test_hourly_mlp_model.py
    │       │   └── train_hourly_mlp_model.py
    │       └── rbf
    │           ├── test_hourly_rbf_model.py
    │           └── train_hourly_rbf_model.py
    ├── poker_hand
    │   ├── preprocessing_testing.py
    │   └── preprocessing_training.py
    └── utilities.py 


Files Descriptions:
    data - this folder contains all the data files used in the project.

    data/unprocessed - data before being processed.
    data/unprocessed/breast_cancer - contains data sets for the breast cancer problem.
    data/unprocessed/currency_exchange - contains data sets for the currency exchange timeseries problem.
    data/unprocessed/poker_hand - contains data sets for the poker hands problem.


    data/processed - data after being processed.
    data/processed/breast_cancer - contains data sets for the breast cancer problem after being processed.
    data/processed/currency_exchange - contains data sets for the currency exchange time series problem.
    data/processed/poker_hand - contains data sets for the poker hands problem after being processed.



    scripts - This folder contains all scripts files used in the project.

    scripts/utilities.py - this script contain utilities functions used through the project.

    scripts/breast_cancer/breast_cancer_preprocessing.py  - script to preprocess breast cancer file.
    scripts/breast_cancer/evaluating_different_topologies.py - script to evaluate the different topologies
    of neural networks by changing the number of neurons in the hidden layer.
    scripts/breast_cancer/evaluation_on_noisy_data.py : script to evaluate the neural networks on noisy data.
    scripts/mlp/test_mlp_model.py  - to test the mlp neural network model for the cancer problem.
    scripts/mlp/train_mlp_model.py - to train the mlp neural network model for the cancer problem.
    scripts/rbf/test_rbf_model.py  - to test the rbf neural network model for the cancer problem.
    scripts/rbf/train_rbf_model.py - to train the rbf neural network model for the cancer problem.



    scripts/poker_hand/preprocessing_training.py - script to preprocessed the raw training data for the poker hands problem.
    scripts/poker_hand/preprocessing_testing.py - script to preprocess the raw testing data for the poker hands problem.


    scripts/currency_exchange/currency_exchange_preprocessing.py -  script to preprocess currency exchange problem files.
    scripts/daily/daily_evaluating_different_topologies.py  - script to evaluate the different topologies
    of neural networks by changing the number of neurons in the hidden layer.
    scripts/daily/daily_evaluation_on_noisy_data.py - script to evaluate the neural networks on noisy data.

    scripts/daily/mlp/test_daily_mlp_model.py - script to test the mlp network model for the problem of currency exchange rate prediction daily.
    scripts/daily/mlp/train_daily_mlp_model.py - script to train the mlp network model for the problem of currency exchange rate prediction daily.
    scripts/daily/rbf/test_daily_rbf_model.py - script to test the rbf network model for the problem of currency exchange rate prediction daily.
    scripts/daily/rbf/train_daily_rbf_model.py - script to train the rbf network model for the problem of currency exchange rate prediction daily.

    scripts/hourly/mlp/test_hourly_mlp_model.py - script to test the mlp network model for the problem of currency exchange rate prediction hourly.
    scripts/hourly/mlp/train_hourly_mlp_model.py - script to train the mlp network model for the problem of currency exchange rate prediction hourly.
    scripts/hourly/rbf/test_hourly_rbf_model.py - script to test the rbf network model for the problem of currency exchange rate prediction hourly.
    scripts/hourly/rbf/train_hourly_rbf_model.py - script to train the rbf network model for the problem of currency exchange rate prediction hourly.


    models - to store the trained models for each problem.
    models/breast_cancer - to store the trained models for the cancer problem.
    models/currency_exchange - to store the trained models for the currency exchange problem.
    models/poker_hand - to store the trained models for the poker hands problem.


    README - Current file. Contains basic information about the project.




* How it works:

    Preprocessing:

        	In order to apply the preprocessing tasks to the raw data sets the following should be done:

        	1. Run: python scripts/breast_cancer/breast_cancer_preprocessing.py
        		This is going to create two files (training.csv, testing.csv) at data/processed/breast_cancer/ directory

        	2. Run: python scripts/currency_exchange/currencty_preprocessing.py
        		This is going to create 4 files:
        			 at_closing_day_training.csv
        			 at_closing_day_testing.csv
        			 hourly_training.csv
        			 hourly_testing.csv

        			 at data/processed/currency_exchange/ directory
                     
            3. Run: python scripts preprocessingTesting.py
                        AND
                    python scripts preprocessingTraining.py



    Training and Testing:

        - Cancer Problem:
           
            1. Run the scripts:         scripts/breast_cancer/mlp/train_mlp_model.py 
                                AND     scripts/breast_cancer/rbf/train_rbf_model.py 
                to train the mlp and rbf neural network models for the cancer classification problem.
           
            2. Run the scripts:         scripts/breast_cancer/mlp/test_mlp_model.py 
                                AND     scripts/breast_cancer/rbf/test_rbf_model.py 
                to test the mlp and rbf trained neural network models for the cancer classification problem.


        - Poker hands Problem:
           
            1. Run the scripts:         scripts/poker_hand/mlp/train_mlp_model.py 
                                AND     scripts/poker_hand/rbf/train_rbf_model.py 
                to train the mlp and rbf neural network models for the poker hands classification problem.
           
            2. Run the scripts:         scripts/poker_hand/mlp/test_mlp_model.py 
                                AND     scripts/poker_hand/rbf/test_rbf_model.py 
                to test the mlp and rbf trained neural network models for the poker hands classification problem.


        - Currency exchange rate prediction Problem:
           
            1. Run the scripts:         scripts/currency_exchange/daily/mlp/train_mlp_model.py 
                                        scripts/currency_exchange/hourly/mlp/train_mlp_model.py 
                                AND     scripts/currency_exchange/daily/rbf/train_rbf_model.py 
                                        scripts/currency_exchange/hourly/rbf/train_rbf_model.py 
                to train the mlp and rbf neural network models for the daily and hourly  currency exchange rate prediction  problems.
           
            2. Run the scripts:         scripts/currency_exchange/daily/mlp/test_mlp_model.py 
                                        scripts/currency_exchange/hourly/mlp/test_mlp_model.py 
                                AND     scripts/currency_exchange/daily/rbf/test_rbf_model.py 
                                        scripts/currency_exchange/hourly/rbf/test_rbf_model.py 
                to test the trained mlp and rbf neural network models for the daily and hourly  currency exchange rate prediction  problems.

    Experiments:

        - Cancer Problem:
            1. Run the scripts:         scripts/breast_cancer/evaluating_different_topologies.py 
                                AND     scripts/breast_cancer/evaluation_on_noisy_data.py 
                to evaluate the different architecture of  mlp and rbf neural network models for the cancer classification problem and to evaluate the effect of adding noisy data to the testing data.

        - Currency exchange rate prediction Problem:
            1. Run the scripts:         scripts/currency_exchange/daily/evaluating_different_topologies.py 
                                        scripts/currency_exchange/hourly/evaluating_different_topologies.py 
                                AND     scripts/currency_exchange/daily/evaluation_on_noisy_data.py
                                        scripts/currency_exchange/hourly/evaluation_on_noisy_data.py  
                
                to evaluate the different architecture of  mlp and rbf neural network models forc urrency exchange rate prediction  problems and to evaluate the effect of adding noisy data to the testing data.


        - Poker hands Problem:
            1. Run the scripts:         scripts/poker_hand/evaluating_different_topologies.py 
                                AND     scripts/poker_hand/evaluation_on_noisy_data.py 
                to evaluate the different architecture of  mlp and rbf neural network models for the poker hands classification problem and to evaluate the effect of adding noisy data to the testing data.



* Dependencies:
	1. Python 2.7    		  - https://www.python.org/downloads/    and    
								http://sourceforge.net/projects/numpy/files/NumPy/
	2. Python Numpy   		  -	http://www.numpy.org/
	3. Matplotlib			  - http://matplotlib.org/users/installing.html
	4. PyWavelets 			  - http://www.pybytes.com/pywavelets/dev/building_extension.html
    5. PyBrain                - http://pybrain.org/pages/download

	