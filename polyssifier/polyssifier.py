#! /usr/bin/env python
import sys
import argparse
import numpy as np
import pickle as p
from multiprocessing import Manager, Pool
import os
import pandas as pd
from copy import deepcopy
from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold
from sklearn.metrics import (f1_score, confusion_matrix, roc_auc_score,
                             mean_squared_error, r2_score)
import joblib
import time
from sklearn.preprocessing import LabelEncoder
from itertools import starmap
from .poly_utils import (build_classifiers, MyVoter, build_regressors,
                         MyRegressionMedianer)
from .report import Report
import logging
from .logger import make_logger
sys.setrecursionlimit(10000)
logger = make_logger('polyssifier')


def poly(data, label, n_folds=10, scale=True, exclude=[],
         feature_selection=False, save=False, scoring='auc',
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
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.ERROR)

    assert label.shape[0] == data.shape[0],\
        "Label dimesions do not match data number of rows"
    _le = LabelEncoder()
    _le.fit(label)
    label = _le.transform(label)
    n_class = len(np.unique(label))
    logger.info(f'Detected {n_class} classes in label')

    if save and not os.path.exists('poly_{}/models'.format(project_name)):
        os.makedirs('poly_{}/models'.format(project_name))

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
    coefficients = {}
    # !fitted_clfs =
    # pd.DataFrame(columns=classifiers.keys(), index = range(n_folds))

    logger.info('Initialization, done.')

    skf = StratifiedKFold(n_splits=n_folds, random_state=1988, shuffle=True)
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
        coefficients[clf_name] = []
        temp = np.zeros((n_class, n_class))
        temp_pred = np.zeros((data.shape[0], ))
        temp_prob = np.zeros((data.shape[0], ))
        clfs = fitted_clfs[clf_name]
        for n in range(n_folds):
            train_score, test_score, prediction, prob, confusion,\
                coefs, fitted_clf = result.pop(0)
            clfs.append(fitted_clf)
            scores.loc[n, (clf_name, 'train')] = train_score
            scores.loc[n, (clf_name, 'test')] = test_score
            temp += confusion
            temp_prob[kf[n][1]] = prob
            temp_pred[kf[n][1]] = _le.inverse_transform(prediction)
            coefficients[clf_name].append(coefs)

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
    return Report(scores=scores, confusions=confusions,
                  predictions=predictions, test_prob=test_prob,
                  coefficients=coefficients,
                  feature_selection=feature_selection)


def _scorer(clf, X, y):
    '''Function that scores a classifier according to what is available as a
    predict function.
    Input:
    - clf = Fitted classifier object
    - X = input data matrix
    - y = estimated labels
    Output:
    - AUC score for binary classification or F1 for multiclass
    The order of priority is as follows:
    - predict_proba
    - decision_function
    - predict
    '''
    n_class = len(np.unique(y))
    if n_class == 2:
        if hasattr(clf, 'predict_proba'):
            ypred = clf.predict_proba(X)
            try:
                ypred = ypred[:, 1]
            except:
                print('predict proba return shape{}'.format(ypred.shape))

            assert len(ypred.shape) == 1,\
                'predict proba return shape {}'.format(ypred.shape)
        elif hasattr(clf, 'decision_function'):
            ypred = clf.decision_function(X)
            assert len(ypred.shape) == 1,\
                'decision_function return shape {}'.format(ypred.shape)
        else:
            ypred = clf.predict(X)
        score = roc_auc_score(y, ypred)
    else:
        score = f1_score(y, clf.predict(X), average='weighted')
    return score


def fit_clf(args, clf_name, val, n_fold, project_name, save, scoring):
    '''
    Multiprocess safe function that fits classifiers
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
    if save and os.path.isfile(file_name):
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
    if hasattr(clf, 'predict_proba'):
        # For compatibility with different sklearn versions
        yprob = clf.predict_proba(X)
        try:
            yprob = yprob[:, 1]
        except:
            print('predict proba return shape {}'.format(yprob.shape))

    elif hasattr(clf, 'decision_function'):
        yprob = clf.decision_function(X)
        try:
            yprob = yprob[:, 1]
        except:
            print('predict proba return shape {}'.format(yprob.shape))

        assert len(yprob.shape) == 1,\
            'predict proba return shape {}'.format(ypred.shape)

    confusion = confusion_matrix(y, ypred)
    duration = time.time() - start
    logger.info('{0:25} {1:2}: Train {2:.2f}/Test {3:.2f}, {4:.2f} sec'.format(
        clf_name, n_fold, train_score, test_score, duration))

    # Feature importance
    if hasattr(clf, 'steps'):
        temp = clf.steps[-1][1]
    elif hasattr(clf, 'best_estimator_'):
        if hasattr(clf.best_estimator_, 'steps'):
            temp = clf.best_estimator_.steps[-1][1]
        else:
            temp = clf.best_estimator_
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
            confusion,  # confusion matrix
            coefficients,  # Coefficients for feature ranking
            clf)  # fitted clf


def create_polynomial(data, degree):
    '''
    :param data: the data (numpy matrix) which will have its data vectors raised to powers
    :param degree: the degree of the polynomial we wish to predict
    :return: a new data matrix of the specified degree (for polynomial fitting purposes)
    '''

    # First we make an empty matrix which is the size of what we wish to pass through to linear regress
    height_of_pass_through = data.shape[0]
    width_of_pass_through = degree * data.shape[1]
    to_pass_through = np.zeros(
        shape=(height_of_pass_through, width_of_pass_through))

    # These are the width and height of each "exponeneted" matrix
    height_exponential_matrix = data.shape[0]
    width_exponential_matrix = data.shape[1]

    for i in range(degree):
        to_add_in = data ** (i + 1)
        for j in range(height_exponential_matrix):
            for k in range(width_exponential_matrix):
                to_pass_through.itemset(
                    (j, k + i * width_exponential_matrix), (to_add_in.item(j, k)))
    return to_pass_through


def polyr(data, label, n_folds=10, scale=True, exclude=[],
          feature_selection=False, num_degrees=1, save=False, scoring='r2',
          project_name='', concurrency=1, verbose=True):
    '''
    Input
    data         = numpy matrix with as many rows as samples
    label        = numpy vector that labels each data row
    n_folds      = number of folds to run
    scale        = whether to scale data or not
    exclude      = list of classifiers to exclude from the analysis
    feature_selection = whether to use feature selection or not (anova)
    num_degrees = the degree of the polynomial model to fit to the data (default is linear)
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
    if num_degrees != 1:
        polynomial_data = create_polynomial(data, num_degrees)
        return polyr(data=polynomial_data, label=label, n_folds=n_folds, scale=scale, exclude=exclude,
                     feature_selection=feature_selection, num_degrees=1, save=save, scoring=scoring,
                     project_name=project_name, concurrency=concurrency, verbose=verbose)

    assert label.shape[0] == data.shape[0],\
        "Label dimesions do not match data number of rows"

    # If the user wishes to save the intermediate steps and there is not already a polyrssifier models directory then
    # this statement creates one.
    if save and not os.path.exists('polyr_{}/models'.format(project_name)):
        os.makedirs('polyr_{}/models'.format(project_name))

    # Whether or not intermeciate steps will be printed out.
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.ERROR)
    logger.info('Building classifiers ...')

    # The main regressors dictionary
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
    skf = KFold(n_splits=n_folds)  # , random_state=1988)
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

    # This calculated the Median of the predictions of the regressors.
    fitted_regs = pd.DataFrame(fitted_regs)
    scores['Median', 'train'] = np.zeros((n_folds, ))
    scores['Median', 'test'] = np.zeros((n_folds, ))
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
    return Report(scores=scores, confusions=confusions,
                  predictions=predictions, test_prob=test_prob,
                  coefficients=coefficients, scoring=scoring,
                  feature_selection=feature_selection)


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

    # Creates the scoring string to pass into grid search.
    if scoring == 'mse':
        scorestring = 'neg_mean_squared_error'
    elif scoring == 'r2':
        scorestring = 'r2'
    else:
        scorestring = 'r2'

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
            kfold = KFold(n_splits=3)  #, random_state=1988)
            reg = GridSearchCV(reg, val['parameters'], n_jobs=1, cv=kfold,
                               scoring=scorestring)
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
    if hasattr(temp, 'coef_'):
        coefficients = temp.coef_
    elif hasattr(temp, 'feature_importances_'):
        coefficients = temp.feature_importances_
    else:
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
    labelcopy = deepcopy(label)

    logger.info(
        'Starting classification with {} workers'.format(args.concurrency))

    # If there are more than 50 unique labels, then it is most likely a regression problem. Otherwise it is probably
    # a classification problem.
    if(len(np.unique(labelcopy)) > 50):
        report = polyr(data, label, n_folds=5, project_name=args.name,
                       concurrency=int(args.concurrency))
    else:
        report = poly(data, label, n_folds=5, project_name=args.name,
                      concurrency=int(args.concurrency))
    report.plot_scores(os.path.join('polyr_' + args.name, args.name))
    report.plot_features(os.path.join('polyr_' + args.name, args.name))
