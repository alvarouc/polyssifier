#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
This module computes the baseline results by applying various
ML classifiers such as SVM, LDA, Naive bayes, K-NN.
"""

# We may want to load memmaps eventually, so here's a flag to control this.

USEJOBLIB=False

import argparse

import functools
from glob import glob
import logging

if USEJOBLIB:
    from joblib.pool import MemmapingPool as Pool
    from joblib.pool import ArrayMemmapReducer as Array
else:
    from multiprocessing import Pool
    from multiprocessing import Array

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as pl
import multiprocessing
import numpy as np
import os
from os import path
import pandas as pd
import pickle
import random as rndc
from scipy.io import savemat
from scipy.spatial.distance import pdist
import seaborn as sb

from sklearn.metrics import auc
from sklearn.metrics import roc_curve as rc
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.lda import LDA
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.qda import QDA

import sys


logging.basicConfig(format="[%(module)s:%(levelname)s]:%(message)s")
logger = logging.getLogger(__name__)

# please set this number to no more than the number of cores on the machine you're
# going to be running it on but high enough to help the computation
PROCESSORS=20
seed = rndc.SystemRandom().seed()
NAMES = ["Nearest Neighbors", "Linear SVM", "RBF SVM",  "Decision Tree",
         "Random Forest", "Logistic Regression", "Naive Bayes", "LDA"]

def make_classifiers(data_shape, ksplit):
    """
    Function that makes classifiers each with a number of folds.

    Parameters
    ---------------
    data_shape: tuple of ints
        Shape of the data.  Must be a pair of integers.
    ksplit: int
        Number of folds.

    Returns:
    classifiers, params
        classifiers is the dictionary of classifiers to be used.
        params is a dictionary of list of dictionaries of the corresponding
        params for each classifier.
    """

    assert len(data_shape) == 2, "Only 2-d data allowed (samples by dimension)."

    classifiers = {
        "Nearest Neighbors": KNeighborsClassifier(3),
        "Linear SVM": SVC(kernel='linear', C=1, probability=True),
        "RBF SVM": SVC(gamma=2, C=1, probability=True),
        "Decision Tree": DecisionTreeClassifier(max_depth=None,
                                                max_features='auto'),
        "Random Forest": RandomForestClassifier(max_depth=None,
                                                n_estimators=10,
                                                max_features='auto',
                                                n_jobs=PROCESSORS/ksplit),
        "Logistic Regression": LogisticRegression(),
        "Naive Bayes": GaussianNB(),
        "LDA": LDA()}

    params = {
        "Nearest Neighbors": [{'n_neighbors': [1, 5, 10, 20]}],
        "Linear SVM": [{'kernel': ['linear'],'C': [1]}],
        "RBF SVM": [{'kernel': ['rbf'],
                     'gamma': np.arange(0.1,1,0.1).tolist()+range(1,10),
                     'C': np.logspace(-2,2,5).tolist()}],
        "Decision Tree": [],
        "Random Forest": [{'n_estimators': range(5,20)}],
        "Logistic Regression": [{'C': np.logspace(0.1,3,7).tolist()}],
        "Naive Bayes": [],
        "LDA": [{'n_components': [np.int(0.1*data_shape[0]),
                                  np.int(0.2*data_shape[0]),
                                  np.int(0.3*data_shape[0]),
                                  np.int(0.5*data_shape[0]),
                                  np.int(0.7*data_shape[0])]}],
        }

    logger.info("Using classifiers %r with params %r" % (classifiers, params))
    return classifiers, params

def get_score(data, labels, fold_pairs,
              name, model, param):
    """
    Function to get score for a classifier.

    Parameters
    ----------
    data: array-like
        Data from which to derive score.
    labels: array-like or list.
        Corresponding labels for each sample.
    fold_pairs: list of pairs of array-like
        A list of train/test indicies for each fold
        (Why can't we just use the KFold object?)
    name: string
        Name of classifier.
    model: WRITEME
    param: WRITEME
        Parameters for the classifier.
    """
    assert isinstance(name, str)
    logger.info("Classifying %s" % name)

    ksplit = len(fold_pairs)
    if name not in NAMES:
        raise ValueError("Classifier %s not supported. "
                         "Did you enter it properly?" % name)

    # Redefine the parameters to be used for RBF SVM (dependent on
    # training data)

    if True:  #better identifier here
        logger.info("Attempting to use grid search...")
        fScore = []
        for i, fold_pair in enumerate(fold_pairs):
            print ("Classifying a %s the %d-th out of %d folds..."
                   % (name, i+1, len(fold_pairs)))
            classifier = get_classifier(name, model, param, data[fold_pair[0], :])
            area = classify(data, labels, fold_pair, classifier)
            fScore.append(area)
    else:
        warnings.warn("Multiprocessing splits not tested yet.")
        pool = Pool(processes=min(ksplit, PROCESSORS))
        classify_func = lambda f : classify(
            data,
            labels,
            fold_pairs[f],
            classifier=get_classifier(
                name,
                model,
                param,
                data=data[fold_pairs[f][0], :]))
        fScore = pool.map(functools.partial(classify_func, xrange(ksplit)))
        pool.close()
        pool.join()

    return classifier, fScore

def get_classifier(name, model, param, data=None):
    """
    Returns the classifier for the model.

    Parameters
    ----------
    WRITEME

    Returns
    -------
    WRITEME
    """
    assert isinstance(name, str)

    if name == "RBF SVM":
        logger.info("RBF SVM requires some preprocessing. "
                    "This may take a while")
        assert data is not None
        #Euclidean distances between samples
        dist = pdist(data, 'euclidean').ravel()
        #Estimates for sigma (10th, 50th and 90th percentile)
        sigest = np.asarray(np.percentile(dist,[10,50,90]))
        #Estimates for gamma (= -1/(2*sigma^2))
        gamma = 1./(2*sigest**2)
        #Set SVM parameters with these values
        param = [{"kernel": ['rbf'],
                  "gamma": gamma.tolist(),
                  "C": np.logspace(-2,2,5).tolist()}]
    if name not in ["Decision Tree", "Naive Bayes"]:
        # why 5?
        logger.info("Using grid search for %s" % name)
        model = GridSearchCV(model, param, cv=5, scoring="f1",
                             n_jobs=PROCESSORS)
    else:
        logger.info("Not using grid search for %s" % name)
    return model

def classify(data, labels, (train_idx, test_idx), classifier=None):
    """
    Classifies given a fold and a model.

    Parameters
    ----------
    WRITEME

    Returns
    -------
    WRITEME
    """

    assert classifier is not None, "Why would you pass not classifier?"

    classifier.fit(data[train_idx, :], labels[train_idx])

    fpr, tpr, thresholds = rc(labels[test_idx],
                              classifier.predict_proba(data[test_idx, :])[:, 1])

    return auc(fpr, tpr)

def load_data(source_dir, data_pattern):
    """
    Loads the data from multiple sources if provided.
    """
    logger.info("Loading data from %s with pattern %s"
                % (source_dir, data_pattern))
    data_files = glob(path.join(source_dir, data_pattern))
    if len(data_files) == 0:
        raise ValueError("No data files found with pattern %s in %s"
                         % (data_pattern, source_dir))

    data = None
    for data_file in data_files:
        d = np.load(data_file)
        if data is not None:
            assert d.shape[0] == data.shape[0]
            data = np.concatenate((data, d), axis=1)
        else:
            data = d

    logger.info("Data loading complete. Shape is %r" % (data.shape,))
    return data

def load_labels(source_dir, label_pattern):
    """
    Function to load labels file.

    Parameters
    ----------
    source_dir: string
        Source directory of labels
    label_pattern: string
        unix regex for label files.

    Returns
    -------
    labels: array-like
        A numpy vector of the labels.
    """

    logger.info("Loading labels from %s with pattern %s"
                % (source_dir, label_pattern))
    label_files = glob(path.join(source_dir, label_pattern))
    if len(label_files) == 0:
        raise ValueError("No label files found with pattern %s"
                         % data_pattern)
    if len(label_files) > 1:
        raise ValueError("Only one label file supported ATM.")
    labels = np.load(label_files[0]).flatten()
    logger.info("Label loading complete. Shape is %r" % (labels.shape,))
    return labels

def main(source_dir, ksplit, out_dir, data_pattern, label_pattern):
    # Load input and labels.
    data = load_data(source_dir, data_pattern)
    labels = load_labels(source_dir, label_pattern)

    # Get classifiers and params.
    classifiers, params = make_classifiers(data.shape, ksplit)

    # preprocess dataset, split into training and test part.
    X, y = data, labels
    X = StandardScaler().fit_transform(X)

    # Make the folds.
    logger.info("Making %d folds" % ksplit)
    kf = StratifiedKFold(labels,
                         n_folds=ksplit)


    # Extract the training and testing indices from the k-fold object,
    # which stores fold pairs of indices.
    fold_pairs = [(tr, ts) for (tr, ts) in kf]
    assert len(fold_pairs) == ksplit

    score={}
    dscore=[]
    for name in NAMES:
        mdl = classifiers[name]
        param = params[name]
        # Get the scores.
        clf, fScores = get_score(data, labels,
                                fold_pairs, name,
                                mdl, param)

        if out_dir is not None:
            with open(path.join(out_dir, name + "%.2f.pkl" % (np.mean(fScores))), "wb") as f:
                pickle.dump(clf,f)

        dscore.append(fScores)
        score[name] = (np.mean(fScores), np.std(fScores))

    dscore = np.asarray(dscore)

    #plot bar charts for the various classifiers
    pl.figure(figsize=[10,6])
    ax=pl.gca()
    sb.barplot(np.array(NAMES), dscore, palette="Paired")
    ax.set_xticks(np.arange(len(NAMES)))
    ax.set_xticklabels(NAMES, rotation=30)
    ax.set_ylabel('classification AUC')
    ax.set_title('Using features: '+str(action_features))
    pl.subplots_adjust(bottom=0.18)
    #pl.draw()
    if out_dir is not None:
        # change the file you're saving it to
        pl.savefig(path.join(out_dir, "classifiers.png"))
    else:
        pl.show(True)

def make_argument_parser():
    """
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("data_directory", help="Directory where the data files live.")
    parser.add_argument("out", help="Output directory of files.")
    parser.add_argument("--folds", default=10, help="Number of folds for n-fold cross validation")
    parser.add_argument("--data_pattern", default="data.npy", help="Pattern for data files")
    parser.add_argument("--label_pattern", default="labels.npy", help="Pattern for label files")
    return parser

if __name__ == "__main__":
    CPUS = multiprocessing.cpu_count()
    if CPUS < PROCESSORS: raise ValueError("Number of PROCESSORS exceed available CPUs, please edits this in the script and come again!")

    parser = make_argument_parser()
    args = parser.parse_args()
    logger.setLevel(logging.DEBUG)
    main(args.data_directory, out_dir=args.out, ksplit=int(args.folds),
         data_pattern=args.data_pattern, label_pattern=args.label_pattern)
