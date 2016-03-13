#! /usr/bin/env python
import sys
import argparse
import numpy as np
import pickle as p
import multiprocessing
from multiprocessing import Manager, Pool
import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
import collections
from copy import deepcopy
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
# logger = multiprocessing.get_logger()
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


def poly(data, label, n_folds=10, scale=True, verbose=True,
         exclude=[], feature_selection=False, save=True, scoring='auc',
         project_name='', concurrency=1):

    data = data.astype(np.float)
    label = label.astype(np.int)
    _le = LabelEncoder()
    label = _le.fit_transform(label)

    if not verbose:
        logger.setLevel(logging.ERROR)
    logger.info('Building classifiers ...')
    classifiers = collections.OrderedDict()
    classifiers['Multilayer Perceptron'] = {
        'clf': MLP(verbose=0, patience=500, learning_rate=1,
                   n_hidden=10, n_deep=2, l1_norm=0, drop=0),
        'parameters': {}}
    classifiers['Nearest Neighbors'] = {
        'clf': KNeighborsClassifier(3),
        'parameters': {'n_neighbors': [1, 5, 10, 20]}}
    #classifiers['SVM'] = {
    #    'clf': SVC(C=1, probability=True, cache_size=10000,
    #               class_weight='balanced'),
    #    'parameters': {'kernel': ['rbf', 'poly'],
    #                   'C': [0.01, 0.1, 1]}}
    classifiers['Linear SVM'] = {
        'clf': LinearSVC(dual=False, class_weight='balanced'),
        'parameters': {'C': [0.01, 0.1, 1],
                       'penalty': ['l1', 'l2']}}
    classifiers['Decision Tree'] = {
        'clf': DecisionTreeClassifier(max_depth=None,
                                      max_features='auto'),
        'parameters': {}}
    classifiers['Random Forest'] = {
        'clf': RandomForestClassifier(max_depth=None,
                                      n_estimators=10,
                                      max_features='auto'),
        'parameters': {'n_estimators': list(range(5, 20))}}
    classifiers['Logistic Regression'] = {
        'clf': LogisticRegression(fit_intercept=True, solver='lbfgs',
                                  penalty='l2'),
        'parameters': {'C': [0.001, 0.1, 1]}}
    classifiers['Naive Bayes'] = {
        'clf': GaussianNB(),
        'parameters': {}}
    #classifiers['Voting'] = {}

    # Remove classifiers that want to be excluded
    for key in exclude:
        if key in classifiers:
            del classifiers[key]

    n_class = len(np.unique(label))

    scores = pd.DataFrame(columns=pd.MultiIndex.from_product(
        [classifiers.keys(), ['train', 'test']]),
                          index=range(n_folds))
    predictions = pd.DataFrame(columns=classifiers.keys(),
                               index=range(data.shape[0]))
    confusions = {}
    fitted_clfs = pd.DataFrame(columns=classifiers.keys(),
                               index=range(n_folds))

    if not os.path.exists('{}_models'.format(project_name)):
        os.makedirs('{}_models'.format(project_name))

    if scale:
        sc = StandardScaler()
        data = sc.fit_transform(data)

    if feature_selection:
        anova_filter = SelectKBest(f_regression, k='all')
        temp = int(np.round(data.shape[1]/5))
        name = lambda x: x['clf']._final_estimator.__class__.__name__.lower()
        for key, val in classifiers.items():
            classifiers[key]['clf'] = make_pipeline(
                anova_filter, classifiers[key]['clf'])
            new_dict = {}
            for keyp in classifiers[key]['parameters']:
                new_dict[name(classifiers[key])+'__'+keyp]\
                    = classifiers[key]['parameters'][keyp]
            classifiers[key]['parameters'] = new_dict
            classifiers[key]['parameters']['selectkbest__k']\
                = np.arange(temp, data.shape[1]-temp, temp).tolist()

    logger.info('Initialization, done.')

    kf = list(StratifiedKFold(label, n_folds=n_folds, random_state=1988))
    manager = Manager()
    args = manager.list()
    args.append({})  # Store inputs
    shared = args[0]
    shared['kf'] = kf
    shared['X'] = data
    shared['y'] = label
    args[0] = shared

    pool = Pool(processes=concurrency)

    args2 = []
    for clf_name, val in classifiers.items():
        for n_fold in range(n_folds):
            args2.append((args, clf_name, val, n_fold, project_name,
                          save, scoring))
    result = pool.starmap(fit_clf, args2)
    pool.close()

    for clf_name in classifiers:
        temp = np.zeros((n_class, n_class))
        temp_pred = np.zeros((data.shape[0], ))
        for n in range(n_folds):
            train_score, test_score, prediction, confusion = result.pop(0)
            scores.loc[n % n_folds, (clf_name, 'train')] = train_score
            scores.loc[n % n_folds, (clf_name, 'test')] = test_score
            temp += confusion
            temp_pred[kf[n % n_folds][1]] = _le.inverse_transform(prediction)

        confusions[clf_name] = temp
        predictions[clf_name] = temp_pred

    # saving confusion matrices
    with open('confusions.pkl', 'wb') as f:
        p.dump(confusions, f, protocol=2)
    return scores, confusions, predictions


def _scorer(clf, X, y):
    score = roc_auc_score(y, clf.predict(X))
    return score


def fit_clf(args, clf_name, val, n_fold, project_name, save, scoring):
    '''
    Run fit method from val with X and y
    clf_name is a string with the classifier name
    '''
    train, test = args[0]['kf'][n_fold]
    X = args[0]['X'][train, :]
    y = args[0]['y'][train]
    file_name = '{}_models/{}_{}.p'.format(project_name, clf_name, n_fold+1)
    start = time.time()
    if os.path.isfile(file_name):
        logger.info('Loading {} {}'.format(file_name, n_fold))
        clf = joblib.load(file_name)
    else:
        logger.info('Training {} {}'.format(clf_name, n_fold))
        clf = deepcopy(val['clf'])
        if val['parameters']:
            clf = GridSearchCV(clf, val['parameters'], n_jobs=1, cv=3,
                               scoring=_scorer)
        clf.fit(X, y)
        if save:
            joblib.dump(clf, file_name)

    train_score = _scorer(clf, X, y)

    X = args[0]['X'][test, :]
    y = args[0]['y'][test]
    # Scores
    test_score = _scorer(clf, X, y)
    ypred = clf.predict(X)
    confusion = confusion_matrix(y, ypred)
    duration = time.time()-start
    logger.info('{0:25} {1:2}:  Train {2:.2f}/ Test {3:.2f}, {4:.2f} sec'.format(clf_name, n_fold, train_score, test_score, duration))
    return (train_score, test_score, ypred, confusion)


def plot(scores, file_name='temp', min_val=None):

    df = scores.apply(np.mean).unstack().join(
        scores.apply(np.std).unstack(), lsuffix='_mean', rsuffix='_std')
    df.columns = ['Test score', 'Train score', 'Test std', 'Train std']
    df.sort_values('Test score', ascending=False, inplace=True)
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
    plt.savefig(file_name + '.svg', transparent=False)

    print(scores)
    return (ax1)

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
    parser.add_argument('--name', default='default',
                        help='Experiment name')
    parser.add_argument('--concurrency', default='1',
                        help='Experiment name')

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

    scores, confusions, predictions = poly(data, label, n_folds=5,
                                           project_name=args.name,
                                           concurrency=int(args.concurrency))
    plot(scores, args.data_directory + args.name)


