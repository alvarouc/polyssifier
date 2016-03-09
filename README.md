[![Coverage Status](https://coveralls.io/repos/alvarouc/polyssifier/badge.svg?branch=master&service=github)](https://coveralls.io/github/alvarouc/polyssifier?branch=master)
[![Build Status](https://travis-ci.org/alvarouc/polyssifier.svg)](https://travis-ci.org/alvarouc/polyssifier)

polyssifier
===========

Run a multitude of classifiers on your data and get an AUC report

Example:

https://github.com/MRN-Code/polyssifier/blob/master/sample/example.ipynb

The class Poly includes several classifiers:

- Multilayer Perceptron (see mlp.py )
- Nearest Neighbors
- Linear SVM
- RBF SVM
- Decision Tree
- Random Forest
- Logistic Regression
- Naive Bayes
- Voting Classifier

You can exclude some of this classfiers by provind a list of names as follows:
```python
from polyssifier import poly, plot

scores, confusions, predictions = poly(data,label, n_folds=5, exclude=['Multilayer Perceptron'], verbose=1, concurrency=4)
plot(scores)
```
