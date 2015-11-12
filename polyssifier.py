import matplotlib
matplotlib.use('Agg')

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
from sklearn.preprocessing import StandardScaler as sc
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier
from mlp import MLP

import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(format="[%(module)s:%(levelname)s]:%(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

PROCESSORS = int(multiprocessing.cpu_count() * 3 / 4)


def make_voter(estimators, y, voting='hard'):
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
                 scale=True, verbose=10, exclude=[]):
        if not os.path.exists('models'):
            os.makedirs('models')

        self.classifiers = {
            'Multilayer Perceptron': {
                'clf': MLP(verbose=0, patience=100),
                'parameters': {'n_hidden': [10],
                               'n_deep': [2, 3],
                               'l1_norm': [0, 0.001],
                               'drop': [0]}},
            'Nearest Neighbors': {
                'clf': KNeighborsClassifier(3),
                'parameters': {'n_neighbors': [1, 5, 10, 20]}},
            'Linear SVM': {
                'clf': SVC(kernel='linear',
                           C=1, probability=True),
                'parameters': {'kernel': ['linear'],
                               'C': [0.01, 0.1, 1]}},
            'RBF SVM': {
                'clf': SVC(gamma=2, C=1, probability=True),
                'parameters': {'kernel': ['rbf'],
                               'gamma': [0.1, 0.5, 1, 5],
                               'C': [0.001, 0.01, 0.1]}},
            'Decision Tree': {
                'clf': DecisionTreeClassifier(max_depth=None,
                                              max_features='auto'),
                'parameters': None},
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
                'parameters': None},
        }

        # Remove classifiers that want to be excluded
        for key in exclude:
            if key in self.classifiers:
                del self.classifiers[key]
        self.exclude = exclude

        self.n_folds = n_folds
        self.scale = scale
        self.label = label
        self.n_class = len(np.unique(label))
        self.verbose = verbose
        self.data = data
        self.scores = {}

    def run(self):

        kf = StratifiedKFold(self.label, n_folds=self.n_folds,
                             random_state=1988)
        for key in self.classifiers:
                self.scores[key] = []
        self.scores['Hard Voting'] = []
        self.scores['Soft Voting'] = []

        for n, (train, test) in enumerate(kf):

            logger.info('Fold {}'.format(n+1))

            X_train, y_train = self.data[train, :], self.label[train]
            X_test, y_test = self.data[test, :], self.label[test]
            estimators = []
            for key, val in self.classifiers.items():

                file_name = 'models/{}_{}.p'.format(key, n+1)
                if os.path.isfile(file_name):
                    logger.info('Loading {}'.format(file_name))
                    with open(file_name, 'rb') as fid:
                        clf = pickle.load(fid)
                else:
                    logger.info('Running {}'.format(key))
                    if val['parameters']:
                        clf = GridSearchCV(val['clf'], val['parameters'],
                                           n_jobs=PROCESSORS, cv=5)
                    else:
                        clf = val['clf']
                    clf.fit(X_train, y_train)
                    with open(file_name, 'wb') as fid:
                        pickle.dump(clf, fid)

                if self.n_class == 2:
                    average = 'binary'
                else:
                    average = 'weighted'

                self.scores[key].append(f1_score(y_test,
                                                 clf.predict(X_test),
                                                 average=average))
                estimators.append((key, clf))
                logger.info('{}_{} : {}'.format(key, n+1,
                                                self.scores[key][-1]))

            # build the voting classifier
            logger.info('Running Voting Classifier')
            clf_hard = make_voter(estimators, y_train, 'hard')
            clf_soft = make_voter(estimators, y_train, 'soft')

            self.scores['Hard Voting']\
                .append(f1_score(y_test, clf_hard.predict(X_test),
                                 average=average))
            logger.info('{}_{} : {}'.format('Hard Voting', n+1,
                                            self.scores['Hard Voting'][-1]))
            self.scores['Soft Voting']\
                .append(f1_score(y_test, clf_soft.predict(X_test),
                                 average=average))
            logger.info('{}_{} : {}'.format('Soft Voting', n+1,
                                            scores['Soft Voting'][-1]))

        return self.scores

    def plot(self, file_name='temp'):

        fig = plt.figure(figsize=[10, 6])
        df = pd.DataFrame(self.scores)

        df = df.describe().T
        df.sort('mean', ascending=True, inplace=True)

        df.plot(kind='barh', y='mean', xerr='std', legend=False,
                color=(0.2, 0.2, 0.7), fontsize=14, width=0.85, alpha=0.7),
        plt.xlim(np.max([(df['mean'] - df['std']).min() - 0.05, 0]), 1)
        plt.title('Classifiers Ranking')

        fig.subplots_adjust(bottom=0.2)
        plt.savefig(file_name + '.pdf')


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
