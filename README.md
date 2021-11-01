
# CS-433 Project 1
## Requirements
The code is tested on Ubuntu 18.04, Python3.8.8 and only Numpy 1.19.5 required
## File structure
- `proj1_helpers.py`: Module containing helper functions:
	- `load_csv_data`: read data from csv files
	- `create_csv_submission`: create output file in csv format
	- `predict_labels`: predict labels using learned weights
	- `acc_score`, `f1_score`: calculate accuracy score and f1 score, respectively
- `preprocessing.py`: Module that contains functions and classes to preprocessing data
	- `Normalizer`: scale data to zero mean and unit variance
	- `Imputer`: handling missing values by replacing or dropping them
	- `PolynomialFeature`: augment feautres using polynominal expansion
	- `NonLinearTransformer`:generate augmented features using arbitrary functions
	- `Pipeline`: preprocessing pipeline
- `implementations.py`: Module that contains six fundamental learning models
	- `least_squares`:least squares using normal equation
    - `least_squares_GD`:least squares using gradient descent
    - `least_squares_SGD`:least squares using stochastic gradient descent
    - `ridge_regression`: ridge regression (l2 regularization)
    - `logistic_regression`: logistic regresssion using gradient descent
    - `reg_logistic_regression`: regularized logistic regresssion (l1 or l2 regularization) using gradient descent
    - and serveral functions used to compute losses and gradients
- `crossval.py`: Module that contains classes used for cross-validation, split the data and grid search
	- `CrossVal`: Regular k-fold crossvalidation
	- `PartitionCrossVal`: for each fold, split the data by `PRI_jet_num`, then train 4 models corresponding to 4 subsets of 4 `PRI_jet_num` values
	- `MassPartitionCrossVal`: for each fold, split the data by `DER_mass_MMC`, then train 2 models corresponding to 2 subsets of whether `DER_mass_MMC` is defined or not
	-  `MultiPartitionCrossVal`: for each fold, split the data by `DER_mass_MMC` and `DER_mass_MMC`, then train 8 models corresponding to 8 combination subsets
	- `GridSearchCV`: using grid search to find the best hyperparameters
- `example.ipynb` Example of various implementation demonstrated in the report
- `run.py`: The training script that obtained the best model and then create the submission file.
    
For more details, you can read the docstrings of each one provided in the code and see the implementatons in `example.ipynb`.

## Reproduce our results
- Download and extract the data then put two csv files into the `data` directory
- Then run the code in root directory of the code: `python3 run.py`
- The result file will be in `result` directory
