#! /usr/bin/env python
import sys
import argparse
import numpy as np
import pickle as p
from multiprocessing import Manager, Pool
import logging
import os
import pandas as pd
from copy import deepcopy
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict, KFold
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, mean_squared_error, r2_score
from sklearn.externals import joblib
import time
from sklearn.preprocessing import LabelEncoder
from itertools import starmap
from .poly_utils import build_classifiers, MyVoter, build_regressors, getRegressors, MyRegressionAverager, \
    MyRegressionMedianer
from .report import Report
sys.setrecursionlimit(10000)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def polyr(data, label, n_folds=10, scale=True, exclude=[],
         feature_selection=False, save=True, scoring='r2',
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
    scoring      = Type of score to use ['mse', 'r2']
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

    #If the user wishes to save the intermediate steps and there is not already a polyrssifier models directory then
    #this statement creates one.
    if save and not os.path.exists('polyr_{}/models'.format(project_name)):
        os.makedirs('polyr_{}/models'.format(project_name))

    #Whether or not intermeciate steps will be printed out.
    if not verbose:
        logger.setLevel(logging.ERROR)
    logger.info('Building classifiers ...')

    #The main regressors dictionary
    regressors = build_regressors(exclude, scale,
                                    feature_selection,
                                    data.shape[1])

    scores = pd.DataFrame(columns=pd.MultiIndex.from_product(
        [regressors.keys(), ['train', 'test']]),
        index=range(n_folds))
    predictions = pd.DataFrame(columns=regressors.keys(),
                               index=range(data.shape[0]))
    test_prob = pd.DataFrame(columns=regressors.keys(),
                             index=range(data.shape[0]))
    confusions = {}
    coefficients = {}
    # !fitted_regs =
    # pd.DataFrame(columns=regressors.keys(), index = range(n_folds))

    logger.info('Initialization, done.')

    # This provides train/test indices to split data in train/test sets.
    skf = KFold(n_splits=n_folds, random_state=1988)
    skf.get_n_splits(np.zeros(data.shape[0]), label)
    kf = list(skf.split(np.zeros(data.shape[0]), label))

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
    for reg_name, val in regressors.items():
        for n_fold in range(n_folds):
            args2.append((args, reg_name, val, n_fold, project_name,
                          save, scoring))

    if concurrency == 1:
        result = list(starmap(fit_reg, args2))
    else:
        pool = Pool(processes=concurrency)
        result = pool.starmap(fit_reg, args2)
        pool.close()

    fitted_regs = {key: [] for key in regressors}

    # Gather results
    for reg_name in regressors:
        coefficients[reg_name] = []
        temp_pred = np.zeros((data.shape[0], ))
        temp_prob = np.zeros((data.shape[0], ))
        regs = fitted_regs[reg_name]
        for n in range(n_folds):
            train_score, test_score, prediction, prob,\
                coefs, fitted_reg = result.pop(0)
            regs.append(fitted_reg)
            scores.loc[n, (reg_name, 'train')] = train_score
            scores.loc[n, (reg_name, 'test')] = test_score
            temp_prob[kf[n][1]] = prob
            temp_pred[kf[n][1]] = prediction
            coefficients[reg_name].append(coefs)

        predictions[reg_name] = temp_pred
        test_prob[reg_name] = temp_prob

    #This calculated the Median of the predictions of the regressors.
    fitted_regs = pd.DataFrame(fitted_regs)
    scores['Median', 'train'] = np.zeros((n_folds, ))
    scores['Median', 'test'] = np.zeros((n_folds, ))
    temp = np.zeros((n_class, n_class))
    temp_pred = np.zeros((data.shape[0], ))
    for n, (train, test) in enumerate(kf):
        reg = MyRegressionMedianer(fitted_regs.loc[n].values)
        X, y = data[train, :], label[train]
        scores.loc[n, ('Median', 'train')] = _reg_scorer(reg, X, y, scoring)
        X, y = data[test, :], label[test]
        scores.loc[n, ('Median', 'test')] = _reg_scorer(reg, X, y, scoring)
        temp_pred[test] = reg.predict(X)

    predictions['Median'] = temp_pred

    if verbose:
        print(scores.astype('float').describe().transpose()
              [['mean', 'std', 'min', 'max']])
    return Report(scores, confusions, predictions, test_prob, coefficients, scoring)


def _reg_scorer(reg, X, y, scoring):
    '''Function that scores a regressor according to what is available as a
    predict function.
    Input:
    - reg = Fitted regressor object
    - X = input data matrix
    - y = corresponding values to the data matrix
    Output:
    - The mean sqaure error or r squared value for the given regressor and data. The default scoring is
    r squared value.
    '''
    if scoring == 'mse':
        return mean_squared_error(y, reg.predict(X))
    else:
        return r2_score(y, reg.predict(X))

def fit_reg(args, reg_name, val, n_fold, project_name, save, scoring):
    '''
    Multiprocess safe function that fits classifiers
    args: shared dictionary that contains
        X: all data
        y: all labels
        kf: list of train and test indexes for each fold
    reg_name: name of the classifier or regressor model
    val: dictionary with
        reg: sklearn compatible classifier 
        parameters: dictionary with parameters, can be used for grid search
    n_fold: number of folds
    project_name: string with the project folder name to save model
    '''
    train, test = args[0]['kf'][n_fold]
    X = args[0]['X'][train, :]
    y = args[0]['y'][train]
    file_name = 'polyr_{}/models/{}_{}.p'.format(
        project_name, reg_name, n_fold + 1)
    start = time.time()
    if os.path.isfile(file_name):
        logger.info('Loading {} {}'.format(file_name, n_fold))
        reg = joblib.load(file_name)
    else:
        logger.info('Training {} {}'.format(reg_name, n_fold))
        reg = deepcopy(val['reg'])
        if val['parameters']:
            kfold = KFold(n_splits=3, random_state=1988)
            reg = GridSearchCV(reg, val['parameters'], n_jobs=1, cv=kfold,
                               scoring=_reg_scorer(scoring=scoring))
        reg.fit(X, y)
        if save:
            joblib.dump(reg, file_name)

    train_score = _reg_scorer(reg, X, y, scoring)

    X = args[0]['X'][test, :]
    y = args[0]['y'][test]
    # Scores
    test_score = _reg_scorer(reg, X, y, scoring)
    ypred = reg.predict(X)
    yprob = 0

    duration = time.time() - start
    logger.info('{0:25} {1:2}: Train {2:.2f}/Test {3:.2f}, {4:.2f} sec'.format(
        reg_name, n_fold, train_score, test_score, duration))

    # Feature importance
    if hasattr(reg, 'steps'):
        temp = reg.steps[-1][1]
    elif hasattr(reg, 'best_estimator_'):
        if hasattr(reg.best_estimator_, 'steps'):
            temp = reg.best_estimator_.steps[-1][1]
        else:
            temp = reg.best_estimator_
    try:
        if hasattr(temp, 'coef_'):
            coefficients = temp.coef_
        elif hasattr(temp, 'feature_importances_'):
            coefficients = temp.feature_importances_
        else:
            coefficients = None
    except:
        coefficients = None

    return (train_score, test_score,
            ypred, yprob,  # predictions and probabilities
            coefficients,  # Coefficients for feature ranking
            reg)  # fitted reg


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

    report = polyr(data, label, n_folds=5, project_name=args.name,
                  concurrency=int(args.concurrency))
    report.plot_scores(os.path.join('polyr_' + args.name, args.name))
    report.plot_features(os.path.join('polyr_' + args.name, args.name))
