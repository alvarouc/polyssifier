#import matplotlib
#matplotlib.use('Agg')
import sys
sys.setrecursionlimit(10000)
import argparse

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
# from sklearn.lda import LDA
import numpy as np
import multiprocessing
import logging
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score

from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier
from sklearn.externals import joblib
from mlp import MLP
import time

import os
import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(format="[%(module)s:%(levelname)s]:%(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

PROCESSORS = int(multiprocessing.cpu_count() * 3 / 4)


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

    def __init__(self, data, label, n_folds=10,
                 scale=True, verbose=10, exclude=[],
                 feature_selection=True):

        if scale:
            sc = StandardScaler()
            data = sc.fit_transform(data)

        if not os.path.exists('models'):
            os.makedirs('models')

        self.classifiers = {
            'Multilayer Perceptron': {
                'clf': MLP(verbose=0, patience=500, learning_rate=1,
                           n_hidden=50, n_deep=3, l1_norm=0,
                           drop=0),
                'parameters': {}},
            'Nearest Neighbors': {
                'clf': KNeighborsClassifier(3),
                'parameters': {'n_neighbors': [1, 5, 10, 20]}},
            'Linear SVM': {
                'clf': SVC(kernel='linear',
                           C=1, probability=True,
                           cache_size=7000),
                'parameters': {'kernel': ['linear'],
                               'C': [0.01, 0.1, 1]}},
            'RBF SVM': {
                'clf': SVC(gamma=2, C=1, probability=True,
                           cache_size=7000),
                'parameters': {'kernel': ['rbf'],
                               'gamma': [0.1, 0.5, 1, 5],
                               'C': [0.001, 0.01, 0.1]}},
            'Decision Tree': {
                'clf': DecisionTreeClassifier(max_depth=None,
                                              max_features='auto'),
                'parameters': {}},
            'Random Forest': {
                'clf': RandomForestClassifier(max_depth=None,
                                              n_estimators=10,
                                              max_features='auto'),
                'parameters': {'n_estimators': list(range(5, 20))}},
            'Logistic Regression': {
                'clf': LogisticRegression(),
                'parameters': {'C': np.logspace(0.1, 3, 5).tolist()}},
            'Naive Bayes': {
                'clf': GaussianNB(),
                'parameters': {}},
        }

        # Remove classifiers that want to be excluded
        for key in exclude:
            if key in self.classifiers:
                del self.classifiers[key]
        self.exclude = exclude

        if feature_selection:
            anova_filter = SelectKBest(f_regression, k='all')
            temp = int(np.round(data.shape[1]/5))
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
                    = np.arange(temp, data.shape[1]-temp, temp).tolist()

        self.n_folds = n_folds
        self.scale = scale
        self.label = LabelEncoder().fit_transform(label)
        self.n_class = len(np.unique(label))
        self.verbose = verbose
        self.data = data
        self.scores = {}

        for key in self.classifiers:
                self.scores[key] = {'train': [], 'test': []}
        self.scores['Hard Voting'] = {'train': [], 'test': []}
        self.scores['Soft Voting'] = {'train': [], 'test': []}

    def fit(self, X, y, n=0):
        # Fits data on all classifiers
        # Checks if data was already fitted
        if self.n_class == 2:
            average = 'binary'
        else:
            average = 'weighted'

        self.fitted_clfs = {}
        for key, val in self.classifiers.items():
            file_name = 'models/{}_{}.p'.format(key, n+1)
            if os.path.isfile(file_name):
                logger.info('Loading {}'.format(file_name))
                start = time.process_time()
                clf = joblib.load(file_name)
                duration = time.process_time()-start
            else:
                logger.info('Running {}'.format(key))
                if val['parameters']:
                    if key == 'Multilayer Perceptron':
                        njobs = 1
                    else:
                        njobs = PROCESSORS
                    clf = GridSearchCV(val['clf'],
                                       val['parameters'],
                                       n_jobs=njobs, cv=3,
                                       iid=False)
                else:
                    clf = val['clf']
                start = time.process_time()
                clf.fit(X, y)
                duration = time.process_time()-start
                joblib.dump(clf, file_name)

            score = f1_score(y, clf.predict(X), average=average)
            self.scores[key]['train'].append(score)

            self.fitted_clfs[key] = clf
            logger.info(
                '{0:25}:  Train {1:.2f}, {2:.2f} sec'.format(
                    key, score, duration))

        # build the voting classifier
        logger.info('Running Voting Classifier')
        clf_hard = make_voter(self.fitted_clfs, y, 'hard')
        clf_soft = make_voter(self.fitted_clfs, y, 'soft')

        self.fitted_clfs['Hard Voting'] = clf_hard
        self.fitted_clfs['Soft Voting'] = clf_soft

        score = f1_score(y, clf_hard.predict(X), average=average)
        self.scores['Hard Voting']['train'].append(score)
        logger.info('{0:25} : Train {1:.2f}'.format('Hard Voting', score))
        score = f1_score(y, clf_soft.predict(X), average=average)
        self.scores['Soft Voting']['train'].append(score)
        logger.info('{0:25} : Train {1:.2f}'.format('Soft Voting', score))

    def test(self, X, y):
        if self.n_class == 2:
            average = 'binary'
        else:
            average = 'weighted'

        for key, val in self.fitted_clfs.items():
            score = f1_score(y, val.predict(X), average=average)
            self.scores[key]['test'].append(score)
            logger.info('{0:25} : Test {1:.2f}'.format(key, score))

    def run(self):

        kf = StratifiedKFold(self.label, n_folds=self.n_folds,
                             random_state=1988)

        for n, (train, test) in enumerate(kf):

            logger.info('Fold {}'.format(n+1))

            X_train, y_train = self.data[train, :], self.label[train]
            X_test, y_test = self.data[test, :], self.label[test]
            self.fit(X_train, y_train, n)
            self.test(X_test, y_test)
        return self.scores

    def plot(self, file_name='temp'):

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

        ax1 = data.plot(kind='bar', yerr=error, colormap='Blues',
                        figsize=(12, 5))
        ax1.set_xticklabels([])
        for n, rect in enumerate(ax1.patches):
            if n > len(self.classifiers)+1:
                break
            ax1.text(rect.get_x()+rect.get_width()/2., 0.01,
                     data.index[n], ha='center', va='bottom',
                     rotation='90', color='black', fontsize=15)
        ax1.yaxis.grid(True)
        ax1.set_ylim(0, 1)
        plt.savefig(file_name + '.pdf')
        return ax1


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
