from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
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
    classifiers = collections.OrderedDict()

    if 'Multilayer Perceptron' not in exclude:
        from mlp import MLP
        classifiers['Multilayer Perceptron'] = {
            'clf': MLP(verbose=0, patience=100, learning_rate=0.1,
                       n_hidden=50, n_deep=2, l1_norm=0.001, drop=0.2),
            'parameters': {}}

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
