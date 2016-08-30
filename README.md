[![Coverage Status](https://coveralls.io/repos/github/alvarouc/polyssifier/badge.svg?branch=master)](https://coveralls.io/github/alvarouc/polyssifier?branch=master)
[![Build Status](https://travis-ci.org/alvarouc/polyssifier.svg)](https://travis-ci.org/alvarouc/polyssifier)

Polyssifier
===========

Polyssifier runs a multitude of machine learning classifiers on your data. It reports scores, confusion matrices, predictions, and a plot of the scores ranked by classifier performance.

## Installation
```bash
pip install polyssifier
```

## How to use
```python
from polyssifier import poly, plot

data = np.load("/path/to/data.npy")
label = np.load("/path/to/labels.npy")
scores, confusions, predictions, probs = poly(data,label, n_folds=8, verbose=1, concurrency=4)
plot(scores)
```


### Requirements
 - Python 3.3 or higher.
 - Keras
 - Sklearn
 - Numpy
 - Pandas
 - MLP

#### Optional
 - Nvidia GPU
 - CUDA

### Features
 - Cross validated results.
   - Report with f1 score (scoring='f1') or ROC (scoring='auc') 
 - Parallel processing. 
   - Control the number of threads with 'concurrency'.
   - We recommend setting concurrency to half the number of Cores in your system.
 - Support for Nvidia GPUs (MLP only). 
   - Set theano flag "device=gpu".
 - Saves trained models for future use in case of server malfunction. 
   - Set project_name for identifying a experiment.
 - Activate feature selection step setting 
   - feature_selection=True
 - Automatically scales your data with scale=True

Not compatible with Python 2 for the moment. We need a replacement for the "starmap" method in multiprocessing 

Example: on sample/example.ipynb
Example:

https://github.com/MRN-Code/polyssifier/blob/master/sample/example.ipynb

It includes the following classifiers:

- Multilayer Perceptron (see mlp.py )
- Nearest Neighbors
- Linear SVM
- RBF SVM
- Decision Tree
- Random Forest
- Logistic Regression
- Naive Bayes
- Voting Classifier

You can exclude some of this classifiers by providing a list of names as follows:
```python
from polyssifier import poly, plot

scores, confusions, predictions = poly(data,label, n_folds=8, exclude=['Multilayer Perceptron'], verbose=1, concurrency=4)
plot(scores)
```
