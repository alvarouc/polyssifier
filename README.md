[![Coverage Status](https://coveralls.io/repos/github/alvarouc/polyssifier/badge.svg)](https://coveralls.io/github/alvarouc/polyssifier) 
![example workflow](https://github.com/alvarouc/polyssifier/actions/workflows/python-package.yml/badge.svg)

Polyssifier
===========

Polyssifier runs a multitude of machine learning models on data. It reports scores, confusion matrices, predictions, and plots the scores ranked by classifier performance.

## Installation
```bash
pip install polyssifier
```

## How to use
### For classification
```python
from polyssifier import poly
# Load data
data = np.load("/path/to/data.npy")
label = np.load("/path/to/labels.npy")
# Run analysis
report = poly(data,label, n_folds=8)
# Plot results
report.plot_scores()
report.plot_features(ntop=10)
```

### For Regression
```python
from polyssifier import polyr
# Load data
data = np.load("/path/to/data.npy")
target = np.load("/path/to/target.npy")
# Run analysis
report = polyr(data, target, n_folds=8)
# Plot results
report.plot_scores()
report.plot_features(ntop=10)
```

### In the terminal
```bash
poly data.npy label.npy --concurrency 10
```

### Requirements
 - Sklearn
 - Numpy
 - Pandas

### Features
 - Cross validated scores.
   - Report f1 score (scoring='f1') or ROC (scoring='auc') for classification
   - Report MSE or R^2 for regression
 - Feature ranking for compatible models (Logistic Regression, Linear SVM, Random Forest)
 - Parallel processing. 
   - Control the number of threads with 'concurrency'.
   - We recommend setting concurrency to half the number of Cores in your system.
 - Saves trained models for future use in case of server malfunction. 
   - Set project_name for identifying a experiment.
 - Activate feature selection step setting 
   - feature_selection=True
 - Automatically scales your data with scale=True

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

and the following regressors:

- Linear Regression
- Bayesian Ridge
- PassiveAggressiveRegressor
- GaussianProcessRegressor
- Ridge
- Lasso
- Lars
- LassoLars
- OrthogonalMatchingPursuit
- ElasticNet

You can exclude some of this models by providing a list of names as follows:
```python
from polyssifier import poly

report = poly(data,label, n_folds=8,
              exclude=['Multilayer Perceptron'])
```
