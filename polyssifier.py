import sys
import argparse
import numpy as np
import multiprocessing
from multiprocessing import Process, Manager
import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle as p
import collections

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.externals import joblib
from mlp import MLP
import time


sys.setrecursionlimit(10000)
# logging.basicConfig(format="[%(module)s:%(levelname)s]:%(message)s")
logger = multiprocessing.get_logger()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

PROCESSORS = int(multiprocessing.cpu_count() // 2)


def make_voter(estimators, y, voting='hard'):
    estimators = list(estimators.items())
    clf = VotingClassifier(estimators, voting)
    clf.estimators_ = [estim for name, estim in estimators]
    clf.le_ = LabelEncoder()
    clf.le_.fit(y)
    clf.classes_ = clf.le_.classes_
    return clf


def make_argument_parser():
    '''
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory',
                        help='Directory where the data files live.')
    parser.add_argument('data', default='data.npy',
                        help='Data file name')
    parser.add_argument('label', default='labels.npy',
                        help='label file name')
    parser.add_argument('--level', default='info',
                        help='Logging level')

    return parser


class Poly:

    def __init__(self, data, label, n_folds=10, scale=True, verbose=False,
                 exclude=[], feature_selection=False, save=True, scoring='f1',
                 name=''):

        label = label.astype(np.int)

        if not verbose:
            logger.setLevel(logging.ERROR)
        logger.info('Building classifiers ...')
        od = collections.OrderedDict()
        od['Multilayer Perceptron'] = {
            'clf': MLP(verbose=0, patience=500, learning_rate=1,
                       n_hidden=10, n_deep=2, l1_norm=0, drop=0),
            'parameters': {}}
        od['Nearest Neighbors'] = {
            'clf': KNeighborsClassifier(3),
            'parameters': {'n_neighbors': [1, 5, 10, 20]}}
        od['SVM'] = {
            'clf': SVC(C=1, probability=True, cache_size=10000,
                       class_weight='balanced'),
            'parameters': {'kernel': ['rbf', 'poly'],
                           'C': [0.01, 0.1, 1]}}
        od['Linear SVM'] = {
            'clf': LinearSVC(dual=False, class_weight='balanced'),
            'parameters': {'C': [0.01, 0.1, 1],
                           'penalty': ['l1', 'l2']}}
        od['Decision Tree'] = {
            'clf': DecisionTreeClassifier(max_depth=None,
                                          max_features='auto'),
            'parameters': {}}
        od['Random Forest'] = {
            'clf': RandomForestClassifier(max_depth=None,
                                          n_estimators=10,
                                          max_features='auto'),
            'parameters': {'n_estimators': list(range(5, 20))}}
        od['Logistic Regression'] = {
            'clf': LogisticRegression(fit_intercept=False, solver='lbfgs',
                                      penalty='l2', class_weight='balanced'),
            'parameters': {'C': [0.001, 0.1, 1]}}
        od['Naive Bayes'] = {
            'clf': GaussianNB(),
            'parameters': {}}
        od['Voting'] = {}

        self.classifiers = od

        # Remove classifiers that want to be excluded
        for key in exclude:
            if key in self.classifiers:
                del self.classifiers[key]

        self.exclude = exclude
        self.feature_selection = feature_selection
        self.n_folds = n_folds
        self.scale = scale
        self._le = LabelEncoder()
        self.label = self._le.fit_transform(label)
        self.n_class = len(np.unique(label))
        self.data = data
        self.scores = {}
        self.confusions = {}
        self._predictions = {}
        self._test_index = []
        self.predictions = None
        self.save = save
        self.name = name
        # Scoring
        self.scoring = scoring
        self._scorer = roc_auc_score
        if scoring == 'f1':
            average = 'binary' if self.n_class == 2 else 'weighted'
            self._scorer = lambda x, y: f1_score(x, y, average=average)

        if not os.path.exists('{}_models'.format(self.name)):
            os.makedirs('{}_models'.format(self.name))

        if self.scale:
            sc = StandardScaler()
            self.data = sc.fit_transform(self.data)

        zeros = np.zeros((self.n_class, self.n_class))
        for key in self.classifiers:
            self.scores[key] = {'train': [], 'test': []}
            self.confusions[key] = np.copy(zeros)
            self._predictions[key] = []
        logger.info('Initialization, done.')

    def _fit_a_classifier(self, data, key, val, n, fitted_clfs, score):
        '''
        Run fit method from val with X and y
        key is a string with the classifier name
        '''
        X = data['X']
        y = data['y']
        file_name = '{}_models/{}_{}.p'.format(self.name, key, n+1)
        start = time.time()
        if os.path.isfile(file_name):
            logger.info('Loading {}'.format(file_name))
            clf = joblib.load(file_name)
        else:
            logger.info('Running {}'.format(key))
            if key == 'Voting':  # should be the last to run
                self.classifiers['Voting']['clf'] =\
                    make_voter(fitted_clfs, y, 'hard')
                clf = val['clf']
            else:
                clf = val['clf']
                if val['parameters']:
                    njobs = 1 if key == 'Multilayer Perceptron' else PROCESSORS
                    clf = GridSearchCV(val['clf'], val['parameters'],
                                       scoring=self.scoring, n_jobs=njobs,
                                       cv=3)
                clf.fit(X, y)
            if self.save:
                joblib.dump(clf, file_name)
        ypred = clf.predict(X)
        score[key] = self._scorer(y, ypred)
        fitted_clfs[key] = clf
        duration = time.time()-start
        logger.info('{0:25}:  Train {1:.2f}, {2:.2f} sec'.format(
            key, score[key], duration))

        return

    def fit(self, X, y, n=0):
        # Fits data on all classifiers
        # Checks if data was already fitted

        manager = Manager()
        fitted_clfs = manager.dict()
        score = manager.dict()
        data = manager.dict()
        data['X'] = X
        data['y'] = y
        ps = []
        for key, val in self.classifiers.items():
            if key != 'Voting':
                ps.append(Process(target=self._fit_a_classifier,
                                  args=(data, key, val, n, fitted_clfs, score)))
                ps[-1].start()
            else:
                [p.join() for p in ps]
                self._fit_a_classifier(data, key, val, n, fitted_clfs, score)

        self.fitted_clfs = fitted_clfs
        return score

    def _predict(self, data, val, key, score, confusion, predictions):
        X = data['X']
        y = data['y']
        ypred = val.predict(X)
        # Scores
        score[key] = self._scorer(y, ypred)
        # Confusion matrix
        confusion[key] = confusion_matrix(y, ypred)
        # Predictions
        predictions[key] = self._le.inverse_transform(ypred)
        logger.info('{0:25} : Test {1:.2f}'.format(key, score[key]))

    def test(self, X, y):

        manager = Manager()
        data = manager.dict()
        score = manager.dict()
        confusion = manager.dict()
        predictions = manager.dict()
        data['X'] = X
        data['y'] = y

        ps = []
        for key, val in self.fitted_clfs.items():
            ps.append(Process(target=self._predict,
                              args=(data, val, key, score, confusion,
                                    predictions)))
            ps[-1].start()

        [p.join() for p in ps]
        return score, confusion, predictions

    def run(self):

        if self.feature_selection:
            anova_filter = SelectKBest(f_regression, k='all')
            temp = int(np.round(self.data.shape[1]/5))
            name = lambda x:\
                x['clf']._final_estimator.__class__.__name__.lower()
            for key, val in self.classifiers.items():
                self.classifiers[key]['clf'] = make_pipeline(
                    anova_filter, self.classifiers[key]['clf'])
                new_dict = {}
                for keyp in self.classifiers[key]['parameters']:
                    new_dict[name(self.classifiers[key])+'__'+keyp]\
                        = self.classifiers[key]['parameters'][keyp]
                self.classifiers[key]['parameters'] = new_dict
                self.classifiers[key]['parameters']['selectkbest__k']\
                    = np.arange(temp, self.data.shape[1]-temp, temp).tolist()

        kf = StratifiedKFold(self.label, n_folds=self.n_folds,
                             random_state=1988)

        for n, (train, test) in enumerate(kf):

            logger.info('Fold {}'.format(n+1))
            X_train, y_train = self.data[train, :], self.label[train]
            X_test, y_test = self.data[test, :], self.label[test]
            scores = self.fit(X_train, y_train, n)
            for key, val in scores.items():
                self.scores[key]['train'].append(scores[key])
            scores,  confusion, prediction = self.test(X_test, y_test)
            for key, val in scores.items():
                self.scores[key]['test'].append(scores[key])
                self.confusions[key] += confusion[key]
                self._predictions[key].extend(
                    self._le.inverse_transform(prediction[key]))
            self._test_index.extend(test)

        self.predictions = pd.DataFrame(self._predictions)
        self.predictions['index'] = self._test_index
        self.predictions.set_index('index', inplace=True)
        self.predictions.sort_index(inplace=True)

        return self.scores

    def plot(self, file_name='temp', min_val=None):

        df = pd.DataFrame(
            [(key, np.mean(score['train']), np.std(score['train']),
              np.mean(score['test']), np.std(score['test']))
             for key, score in self.scores.items()],
            columns=['classifier', 'Train score',
                     'Train std', 'Test score',
                     'Test std'])

        df.sort_values('Test score', ascending=False, inplace=True)
        df = df.set_index('classifier')
        error = df[['Train std', 'Test std']]
        error.columns = ['Train score', 'Test score']
        data = df[['Train score', 'Test score']]

        nc = df.shape[0]

        ax1 = data.plot(kind='bar', yerr=error, colormap='Blues',
                        figsize=(nc*2, 5), alpha=0.7)
        ax1.set_xticklabels([])
        ax1.set_xlabel('')
        ax1.yaxis.grid(True)

        temp = np.array(data)
        ylim = np.max(temp.min()-.1, 0) if min_val is None else min_val

        ax1.set_ylim(ylim, 1)
        for n, rect in enumerate(ax1.patches):
            if n >= nc:
                break
            ax1.text(rect.get_x()-rect.get_width()/2., ylim + (1-ylim)*.01,
                     data.index[n], ha='center', va='bottom',
                     rotation='90', color='black', fontsize=15)
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                   ncol=2, fancybox=True, shadow=True)
        plt.savefig(file_name + '.pdf')
        plt.savefig(file_name + '.svg', transparent=False,
                    bbox_inches='tight', pad_inches=0)

        # saving confusion matrices
        with open('confusions.pkl', 'wb') as f:
            p.dump(self.confusions, f, protocol=2)

        return (ax1)

if __name__ == '__main__':

    parser = make_argument_parser()
    args = parser.parse_args()

    if args.level == 'info':
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)

    data = np.load(args.data_directory + args.data)
    label = np.load(args.data_directory + args.label)

    logger.info(
        'Starting classification with {} workers'.format(PROCESSORS))

    poly = Poly(data, label, n_folds=5)
    poly.run()
    poly.plot(args.data_directory + args.data)
