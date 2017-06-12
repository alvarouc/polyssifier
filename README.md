[![Coverage Status](https://coveralls.io/repos/github/alvarouc/polyssifier/badge.svg?branch=master)](https://coveralls.io/github/alvarouc/polyssifier?branch=master)
[![Build Status](https://travis-ci.org/alvarouc/polyssifier.svg)](https://travis-ci.org/alvarouc/polyssifier)

Polyssifier
===========

Polyssifier runs a multitude of machine learning classifiers on your data, reports scores, confusion matrices, predictions, and plots the scores ranked by classifier performance.

## Installation
```bash
pip install polyssifier
```

## How to use
```python
from polyssifier import poly
# Load data
data = np.load("/path/to/data.npy")
label = np.load("/path/to/labels.npy")
# Run analysis
report = poly(data,label, n_folds=8, verbose=1, concurrency=4)
# Plot results
report.plot_scores()
report.plot_features(ntop=10)
```

```bash
poly data.npy label.npy --concurrency 10
```

### Requirements
 - Python 3.3 or higher.
 - Keras
 - Sklearn
 - Numpy
 - Pandas

#### Optional
 - Nvidia GPU
 - CUDA

### Features
 - Cross validated scores.
   - Report with f1 score (scoring='f1') or ROC (scoring='auc')
 - Feature ranking for compatible models (Logistic Regression, Linear SVM, Random Forest)
 - Parallel processing. 
   - Control the number of threads with 'concurrency'.
   - We recommend setting concurrency to half the number of Cores in your system.
 - Saves trained models for future use in case of server malfunction. 
   - Set project_name for identifying a experiment.
 - Activate feature selection step setting 
   - feature_selection=True
 - Automatically scales your data with scale=True

Not compatible with Python 2 for the moment. We need a replacement for the "starmap" method in multiprocessing 

Example: on [sample/example.ipynb](sample/example.ipynb)

It includes the following classifiers:

- Multilayer Perceptron
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
from polyssifier import poly

report = poly(data,label, n_folds=8,
              exclude=['Multilayer Perceptron'])
```
