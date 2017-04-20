#! /usr/bin/env python
import sys
import argparse
import numpy as np
import pickle as p
from multiprocessing import Manager, Pool
import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.model_selection import StratifiedKFold
from dask_searchcv import GridSearchCV
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score
from sklearn.externals import joblib
import time
from sklearn.preprocessing import LabelEncoder
from itertools import starmap
from .poly_utils import build_classifiers, MyVoter

sys.setrecursionlimit(10000)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def poly(data, label, n_folds=10, scale=True, exclude=[],
         feature_selection=False, save=True, scoring='auc',
         project_name='', concurrency=1, verbose=True):
    '''
    Input
    data         = numpy matrix with as many rows as samples
    label        = numpy vector that labels each data row
    n_folds      = number of folds to run
    scale        = whether to scale data or not
    exclude      = list of classifiers to exclude from the analysis
    feature_selection = whether to use feature selection or not (anova)
    save         = whether to save intermediate steps or not
    scoring      = Type of score to use ['auc', 'f1']
    project_name = prefix used to save the intermediate steps
    concurrency  = number of parallel jobs to run
    verbose      = whether to print or not results

    Ouput
    scores       = matrix with scores for each fold and classifier
    confusions   = confussion matrix for each classifier
    predictions  = Cross validated predicitons for each classifier
    '''

    assert label.shape[0] == data.shape[0],\
        "Label dimesions do not match data number of rows"
    _le = LabelEncoder()
    _le.fit(label)
    label = _le.transform(label)
    n_class = len(np.unique(label))

    if save and not os.path.exists('poly_{}/models'.format(project_name)):
        os.makedirs('poly_{}/models'.format(project_name))

    if not verbose:
        logger.setLevel(logging.ERROR)
    logger.info('Building classifiers ...')
    classifiers = build_classifiers(exclude, scale,
                                    feature_selection,
                                    data.shape[1])

    scores = pd.DataFrame(columns=pd.MultiIndex.from_product(
        [classifiers.keys(), ['train', 'test']]),
        index=range(n_folds))
    predictions = pd.DataFrame(columns=classifiers.keys(),
                               index=range(data.shape[0]))
    test_prob = pd.DataFrame(columns=classifiers.keys(),
                             index=range(data.shape[0]))
    confusions = {}
    # !fitted_clfs =
    # pd.DataFrame(columns=classifiers.keys(), index = range(n_folds))

    logger.info('Initialization, done.')

    skf = StratifiedKFold(n_splits=n_folds, random_state=1988)
    skf.get_n_splits(np.zeros(data.shape[0]), label)
    kf = list(skf.split(np.zeros(data.shape[0]),label))

    # Parallel processing of tasks
    manager = Manager()
    args = manager.list()
    args.append({})  # Store inputs
    shared = args[0]
    shared['kf'] = kf
    shared['X'] = data
    shared['y'] = label
    args[0] = shared

    args2 = []
    for clf_name, val in classifiers.items():
        for n_fold in range(n_folds):
            args2.append((args, clf_name, val, n_fold, project_name,
                          save, scoring))

    if concurrency == 1:
        result = list(starmap(fit_clf, args2))
    else:
        pool = Pool(processes=concurrency)
        result = pool.starmap(fit_clf, args2)
        pool.close()

    fitted_clfs = {key: [] for key in classifiers}

    # Gather results
    for clf_name in classifiers:
        temp = np.zeros((n_class, n_class))
        temp_pred = np.zeros((data.shape[0], ))
        temp_prob = np.zeros((data.shape[0], ))
        clfs = fitted_clfs[clf_name]
        for n in range(n_folds):
            train_score, test_score, prediction, prob, confusion,\
                fitted_clf = result.pop(0)
            clfs.append(fitted_clf)
            scores.loc[n, (clf_name, 'train')] = train_score
            scores.loc[n, (clf_name, 'test')] = test_score
            temp += confusion
            temp_prob[kf[n][1]] = prob
            temp_pred[kf[n][1]] = _le.inverse_transform(prediction)

        confusions[clf_name] = temp
        predictions[clf_name] = temp_pred
        test_prob[clf_name] = temp_prob

    # Voting
    fitted_clfs = pd.DataFrame(fitted_clfs)
    scores['Voting', 'train'] = np.zeros((n_folds, ))
    scores['Voting', 'test'] = np.zeros((n_folds, ))
    temp = np.zeros((n_class, n_class))
    temp_pred = np.zeros((data.shape[0], ))
    for n, (train, test) in enumerate(kf):
        clf = MyVoter(fitted_clfs.loc[n].values)
        X, y = data[train, :], label[train]
        scores.loc[n, ('Voting', 'train')] = _scorer(clf, X, y)
        X, y = data[test, :], label[test]
        scores.loc[n, ('Voting', 'test')] = _scorer(clf, X, y)
        temp_pred[test] = clf.predict(X)
        temp += confusion_matrix(y, temp_pred[test])

    confusions['Voting'] = temp
    predictions['Voting'] = temp_pred
    test_prob['Voting'] = temp_pred
    ######

    # saving confusion matrices
    if save:
        with open('poly_' + project_name + '/confusions.pkl', 'wb') as f:
            p.dump(confusions, f, protocol=2)

    if verbose:
        print(scores.astype('float').describe().transpose()
              [['mean', 'std', 'min', 'max']])
    return scores, confusions, predictions, test_prob


def _scorer(clf, X, y):
    n_class = len(np.unique(y))
    if n_class == 2:
        if hasattr(clf, 'predict_proba'):
            ypred = clf.predict_proba(X)[:, 1]
        elif hasattr(clf, 'decision_function'):
            ypred = clf.decision_function(X)
        else:
            ypred = clf.predict(X)
        score = roc_auc_score(y, ypred)
    else:
        score = f1_score(y, clf.predict(X))
    return score


def fit_clf(args, clf_name, val, n_fold, project_name, save, scoring):
    '''
    args: shared dictionary that contains 
        X: all data 
        y: all labels
        kf: list of train and test indexes for each fold
    clf_name: name of the classifier model
    val: dictionary with
        clf: sklearn compatible classifier 
        parameters: dictionary with parameters, can be used for grid search
    n_fold: number of folds
    project_name: string with the project folder name to save model
    '''
    train, test = args[0]['kf'][n_fold]
    X = args[0]['X'][train, :]
    y = args[0]['y'][train]
    file_name = 'poly_{}/models/{}_{}.p'.format(
        project_name, clf_name, n_fold + 1)
    start = time.time()
    if os.path.isfile(file_name):
        logger.info('Loading {} {}'.format(file_name, n_fold))
        clf = joblib.load(file_name)
    else:
        logger.info('Training {} {}'.format(clf_name, n_fold))
        clf = deepcopy(val['clf'])
        if val['parameters']:
            clf = GridSearchCV(clf, val['parameters'], cv=3,
                               scoring=_scorer, n_jobs=-1)
        clf.fit(X, y)
        if save:
            joblib.dump(clf, file_name)

    train_score = _scorer(clf, X, y)

    X = args[0]['X'][test, :]
    y = args[0]['y'][test]
    # Scores
    test_score = _scorer(clf, X, y)
    ypred = clf.predict(X)
    if hasattr(clf, 'predict_proba'):
        yprob = clf.predict_proba(X)
    elif hasattr(clf, 'decision_function'):
        yprob = clf.decision_function(X)

    confusion = confusion_matrix(y, ypred)
    duration = time.time() - start
    logger.info('{0:25} {1:2}: Train {2:.2f}/Test {3:.2f}, {4:.2f} sec'.format(
        clf_name, n_fold, train_score, test_score, duration))
    return (train_score, test_score,
            ypred, yprob,  # predictions and probabilities
            confusion,  # confusion matrix
            clf)  # fitted clf


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
                    figsize=(nc * 2, 5), alpha=0.7)
    ax1.set_xticklabels([])
    ax1.set_xlabel('')
    ax1.yaxis.grid(True)

    temp = np.array(data)
    ylim = np.max(temp.min() - .1, 0) if min_val is None else min_val

    ax1.set_ylim(ylim, 1)
    for n, rect in enumerate(ax1.patches):
        if n >= nc:
            break
        ax1.text(rect.get_x() - rect.get_width() / 2., ylim + (1 - ylim) * .01,
                 data.index[n], ha='center', va='bottom',
                 rotation='90', color='black', fontsize=15)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=2, fancybox=True, shadow=True)
    plt.savefig(file_name + '.pdf')
    plt.savefig(file_name + '.svg', transparent=False)
    return (ax1)


def make_argument_parser():
    '''
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('data', default='data.npy',
                        help='Data file name')
    parser.add_argument('label', default='labels.npy',
                        help='label file name')
    parser.add_argument('--level', default='info',
                        help='Logging level')
    parser.add_argument('--name', default='default',
                        help='Experiment name')
    parser.add_argument('--concurrency', default='1',
                        help='Number of allowed concurrent processes')

    return parser

if __name__ == '__main__':

    parser = make_argument_parser()
    args = parser.parse_args()

    if args.level == 'info':
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)

    data = np.load(args.data)
    label = np.load(args.label)

    logger.info(
        'Starting classification with {} workers'.format(args.concurrency))

    scores, confusions, predictions, test_prob = poly(data, label, n_folds=5, project_name=args.name,
                                                      concurrency=int(args.concurrency))
    plot(scores, os.path.join('poly_' + args.name, args.name))
