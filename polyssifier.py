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
from sklearn.metrics import f1_score
from sklearn.grid_search import GridSearchCV


import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

logging.basicConfig(format="[%(module)s:%(levelname)s]:%(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

PROCESSORS = int(multiprocessing.cpu_count() / 2)

CLASSIFIER_PARAMETER = {

    'Nearest Neighbors': {
        'clf': KNeighborsClassifier(3),
        'parameters': {'n_neighbors': [1, 5, 10, 20]}},

    'Linear SVM': {
        'clf': SVC(kernel='linear',
                   C=1, probability=True),
        'parameters': {'kernel': ['linear'],
                       'C': [0.1, 0.25, 0.5, 1]}},

    'RBF SVM': {
        'clf': SVC(gamma=2, C=1, probability=True),
        'parameters': {'kernel': ['rbf'],
                       'gamma': [0.1, 0.5, 1, 5, 10],
                       'C': np.logspace(-2, 2, 5).tolist()}},

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
        'parameters': {'C': np.logspace(0.1, 3, 7).tolist()}},

    'Naive Bayes': {
        'clf': GaussianNB(),
        'parameters': None},
}


class Partition:
        def __init__(self, data, label, n_folds=10):
            self.i = 0
            self.data = data
            self.label = label
            self.n_folds = n_folds
            self.kf = [(train, test) for train, test in
                       StratifiedKFold(label, n_folds=10)]

        def __iter__(self):
            self.i = 0
            return self

        def __next__(self):
            if self.i < self.n_folds:
                logger.info('Working on fold {}'.format(self.i+1))
                train_index, test_index = self.kf[self.i]
                train = {'X': self.data[train_index, :],
                         'y': self.label[train_index]}
                test = {'X': self.data[test_index, :],
                        'y': self.label[test_index]}
                self.i += 1
                return(train, test)
            else:
                raise StopIteration()


class Poly:

    def __init__(self, data, label, n_folds=10):
        self.n_folds = n_folds
        self.data = Partition(data, label, self.n_folds)
        self.results = {}
        if len(np.unique(label)) > 2:
            self.scorer = lambda x, y: \
                f1_score(x, y, average='weighted')
        else:
            self.scorer = lambda x, y: \
                f1_score(x, y, average='binary')

    def cv(self, clf, parameters=None):
        if parameters:
            clf = GridSearchCV(clf, parameters,
                               n_jobs=PROCESSORS,
                               cv=self.n_folds)
        train_s = []
        test_s = []
        for train, test in self.data:
            clf.fit(**train)
            train_p = clf.predict(train['X'])
            test_p = clf.predict(test['X'])
            train_s.append(self.scorer(train['y'], train_p))
            test_s.append(self.scorer(test['y'], test_p))
            logger.info(
                'train/test AUC: {0:.1f}/{1:.1f}]'
                .format(train_s[-1]*100, test_s[-1]*100))
        return(train_s, test_s)

    def run(self):
        for key, value in CLASSIFIER_PARAMETER.items():
            logger.info('Running {}...'.format(key))
            train_s, test_s = self.cv(**value)
            self.results[key] = test_s

        return(self.results)

    def plot(self, file_name='temp'):

        fig = plt.figure(figsize=[10, 6])
        ds = pd.DataFrame(self.results)
        print(ds)
        ds.to_csv(file_name + '_results.csv')
        ds_long = pd.melt(ds)
        sb.barplot(x='variable', y='value',
                   data=ds_long, palette='Paired')
        plt.xticks(rotation=30)
        plt.title('Classification AUC Mean +- SD')
        plt.xlabel('')
        fig.subplots_adjust(bottom=0.2)
        plt.savefig(file_name + '.pdf')


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
