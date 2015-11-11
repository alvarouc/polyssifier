polyssifier
===========

Run a multitude of classifiers on your data and get an AUC report

Example: 

https://github.com/alvarouc/polyssifier/blob/master/sample/example.ipynb

The class Poly includes several classifiers:

- Multilayer Perceptron (see https://github.com/alvarouc/mlp )
- Nearest Neighbors
- Linear SVM
- RBF SVM
- Desicion Tree
- Random Forest
- Logistic Regression
- Naive Bayes
- Voting Classifier

You can exclude some of this classfiers by provind a list of names as follows:
```python
from polyssifier import Poly

pol = Poly(data,label, n_folds=5, exclude=['Multilayer Perceptron'], verbose =1)

scores= pol.run()
```