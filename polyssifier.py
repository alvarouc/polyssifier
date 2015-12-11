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
from sklearn.metrics import f1_score, confussion_matrix

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
                           n_hidden=10, n_deep=2, l1_norm=0,
                           drop=0),
                'parameters': {}},
            'Nearest Neighbors': {
                'clf': KNeighborsClassifier(3),
                'parameters': {'n_neighbors': [1, 5, 10, 20]}},
            'SVM': {
                'clf': SVC(C=1, probability=True,
                           cache_size=10000),
                'parameters': {'kernel': ['linear', 'rbf', 'poly'],
                               'C': [0.01, 0.1, 1]}},
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
                'clf': LogisticRegression(fit_intercept=False,
                                          solver='lbfgs', penalty='l2'),
                'parameters': {'C': [0.001, 0.1, 1]}},
            'Naive Bayes': {
                'clf': GaussianNB(),
                'parameters': {}},
        }

        # Remove classifiers that want to be excluded
        for key in exclude:
            if key in self.classifiers:
                del self.classifiers[key]
        self.exclude = exclude

        self.feature_selection = feature_selection

        self.n_folds = n_folds
        self.scale = scale
        self.label = LabelEncoder().fit_transform(label)
        self.n_class = len(np.unique(label))
        self.verbose = verbose
        self.data = data
        self.scores = {}
        self.confusions = {}

        zeros = np.zeros((self.n_class, self.n_class))
        for key in self.classifiers:
                self.scores[key] = {'train': [], 'test': []}
                self.confussions[key] = {'train': zeros, 'test': zeros}
        self.scores['Voting'] = {'train': [], 'test': []}

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
            start = time.process_time()
            if os.path.isfile(file_name):
                logger.info('Loading {}'.format(file_name))
                clf = joblib.load(file_name)
            else:
                logger.info('Running {}'.format(key))
                if val['parameters']:
                    if key == 'Multilayer Perceptron':
                        njobs = 1
                    else:
                        njobs = PROCESSORS
                    clf = GridSearchCV(val['clf'], val['parameters'],
                                       n_jobs=njobs, cv=3, iid=False)
                else:
                    clf = val['clf']

                clf.fit(X, y)
                joblib.dump(clf, file_name)

            duration = time.process_time()-start

            ypred = clf.predict(X)
            score = f1_score(y, ypred, average=average)
            self.scores[key]['train'].append(score)
            confussion = confussion_matrix(y, ypred)
            self.confusions[key]['train']+=confussion

            self.fitted_clfs[key] = clf
            logger.info('{0:25}:  Train {1:.2f}, {2:.2f} sec'.format(
                key, score, duration))

        # build the voting classifier
        logger.info('Running Voting Classifier')
        clf = make_voter(self.fitted_clfs, y, 'hard')

        self.fitted_clfs['Voting'] = clf

        ypred = clf.predict(X)
        score = f1_score(y, ypred, average=average)
        self.scores['Voting']['train'].append(score)
        confussion = confussion_matrix(y, ypred)
        self.confusions['Voting']['train']+=confussion
        
        logger.info('{0:25} : Train {1:.2f}'.format('Voting', score))

    def test(self, X, y):
        if self.n_class == 2:
            average = 'binary'
        else:
            average = 'weighted'

        for key, val in self.fitted_clfs.items():
            ypred = val.predict(X)
            score = f1_score(y, ypred, average=average)
            self.scores[key]['test'].append(score)
            confussion = confussion_matrix(y, ypred)
            self.confusions[key]['train']+=confussion
            logger.info('{0:25} : Test {1:.2f}'.format(key, score))

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
        print(df)
        error = df[['Train std', 'Test std']]
        error.columns = ['Train score', 'Test score']
        data = df[['Train score', 'Test score']]

        ax1 = data.plot(kind='bar', yerr=error, colormap='Blues',
                        figsize=(12, 5), alpha=0.7)
        ax1.set_xticklabels([])
        ax1.set_xlabel('')
        ax1.yaxis.grid(True)
        ylim = np.max(np.array(data).min()-.1, 0)
        ax1.set_ylim(ylim, 1)
        for n, rect in enumerate(ax1.patches):
            if n > len(self.classifiers):
                break
            ax1.text(rect.get_x()-rect.get_width()/2., ylim + (1-ylim)*.01,
                     data.index[n], ha='center', va='bottom',
                     rotation='90', color='black', fontsize=15)
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                   ncol=2, fancybox=True, shadow=True)
        plt.savefig(file_name + '.pdf')
        
        return (ax1, df)


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
