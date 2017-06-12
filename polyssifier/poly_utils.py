from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, BayesianRidge, Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.gaussian_process import GaussianProcessRegressor
import collections
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class MyVoter(object):
    """Voter that receive fitted classifiers

    """

    def __init__(self, estimators):
        self.estimators_ = estimators

    def predict(self, X):
        predictions = np.asarray(
            [clf.predict(X) for clf in self.estimators_]).T
        maj = np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x)), axis=1,
            arr=predictions.astype('int'))
        return maj


def build_classifiers(exclude, scale, feature_selection, nCols):
    '''
    This method builds an OrderedDict (similar to a map) of classifiers, where the key is the name of the 
    classifiers and the value contains the classifier object itself and some associated parameters.
    '''
    classifiers = collections.OrderedDict()

    if 'Multilayer Perceptron' not in exclude:
        classifiers['Multilayer Perceptron'] = {
            'clf': MLP(),
            'parameters': {'hidden_layer_sizes': [(100, 50), (50, 25)],
                           'max_iter': [500]}
        }

    if 'Nearest Neighbors' not in exclude:
        classifiers['Nearest Neighbors'] = {
            'clf': KNeighborsClassifier(),
            'parameters': {'n_neighbors': [1, 5, 10, 20]}}

    if 'SVM' not in exclude:
        classifiers['SVM'] = {
            'clf': SVC(C=1, probability=True, cache_size=10000,
                       class_weight='balanced'),
            'parameters': {'kernel': ['rbf', 'poly'],
                           'C': [0.01, 0.1, 1]}}

    if 'Linear SVM' not in exclude:
        classifiers['Linear SVM'] = {
            'clf': LinearSVC(dual=False, class_weight='balanced'),
            'parameters': {'C': [0.01, 0.1, 1],
                           'penalty': ['l1', 'l2']}}

    if 'Decision Tree' not in exclude:
        classifiers['Decision Tree'] = {
            'clf': DecisionTreeClassifier(max_depth=None,
                                          max_features='auto'),
            'parameters': {}}

    if 'Random Forest' not in exclude:
        classifiers['Random Forest'] = {
            'clf': RandomForestClassifier(max_depth=None,
                                          n_estimators=10,
                                          max_features='auto'),
            'parameters': {'n_estimators': list(range(5, 20))}}

    if 'Logistic Regression' not in exclude:
        classifiers['Logistic Regression'] = {
            'clf': LogisticRegression(fit_intercept=True, solver='lbfgs',
                                      penalty='l2'),
            'parameters': {'C': [0.001, 0.1, 1]}}

    if 'Naive Bayes' not in exclude:
        classifiers['Naive Bayes'] = {
            'clf': GaussianNB(),
            'parameters': {}}
    # classifiers['Voting'] = {}

    def name(x):
        """
        :param x: The name of the classifier
        :return: The class of the final estimator in lower case form
        """
        return x['clf']._final_estimator.__class__.__name__.lower()

    for key, val in classifiers.items():
        if not scale and not feature_selection:
            break
        steps = []
        if scale:
            steps.append(StandardScaler())
        if feature_selection:
            steps.append(SelectKBest(f_regression, k='all'))
        steps.append(classifiers[key]['clf'])
        classifiers[key]['clf'] = make_pipeline(*steps)
        # Reorganize paramenter list for grid search
        new_dict = {}
        for keyp in classifiers[key]['parameters']:
            new_dict[name(classifiers[key]) + '__' +
                     keyp] = classifiers[key]['parameters'][keyp]
        classifiers[key]['parameters'] = new_dict
        if nCols > 5 and feature_selection:
            classifiers[key]['parameters']['selectkbest__k'] = np.linspace(
                np.round(nCols / 5), nCols, 5).astype('int').tolist()

    return classifiers

def build_regressors(exclude, scale, feature_selection, nCols):
    '''
    This method builds an OrderedDict (similar to a map) of classifiers, where the key is the name of the
    classifiers and the value contains the classifier object itself and some associated parameters.
    '''
    regressors = collections.OrderedDict()

    if 'Linear Regression' not in exclude:
        regressors['Linear Regression'] = {
            'reg': LinearRegression(),
            'parameters': {}
        }

    if 'Bayesian Ridge' not in exclude:
        regressors['Bayesian Ridge'] = {
            'reg': BayesianRidge(),
            'parameters': {}
        }

    if 'Perceptron' not in exclude:
        regressors['Perceptron'] = {
            'reg': Perceptron(),
            'parameters': {}
        }

    if 'GaussianProcessRegressor' not in exclude:
        regressors['GaussianProcessRegressor'] = {
            'reg': GaussianProcessRegressor(),
            'parameters': {}
        }


    return regressors