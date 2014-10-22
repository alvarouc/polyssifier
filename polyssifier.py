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
        classifiers is the set of classifiers to be used.
        params is a list of list of dictionaries (WHY?) of the corresponding params for each classifier.
    """

    assert len(data_shape) == 2, "Only 2-d data allowed (samples by dimension)."

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel='linear', C=1, probability=True),
        SVC(gamma=2, C=1, probability=True),
        DecisionTreeClassifier(max_depth=None, max_features='auto'),
        RandomForestClassifier(max_depth=None,
                               n_estimators=10,
                               max_features='auto',
                               n_jobs=PROCESSORS/ksplit),
        LogisticRegression(),
        GaussianNB(),
        LDA()]
    params = [
        [{'n_neighbors': [1, 5, 10, 20]}],                             # KNN
        [{'kernel': ['linear'],'C': [1]}],                          # linearSVM
        [{'kernel': ['rbf'],
          'gamma': np.arange(0.1,1,0.1).tolist()+range(1,10),
          'C': np.logspace(-2,2,5).tolist()}],                      # rbf SVM
        [],                                                         # decision tree
        [{'n_estimators': range(5,20)}],                            # random forest
        [{'C': np.logspace(0.1,3,7).tolist()}],                     # logistic regression
        [],                                                         # Naive Bayes
        [{'n_components': [np.int(0.1*data_shape[0]),
                           np.int(0.2*data_shape[0]),
                           np.int(0.3*data_shape[0]),
                           np.int(0.5*data_shape[0]),
                           np.int(0.7*data_shape[0])]}],             # LDA
    ]
    return classifiers, params

def get_score(data, labels, idx, ksplit, name, mdl, param, fld):
    """
    Function to get score for a classifier.

    Parameters
    ----------
    data: array-like
        Data from which to derive score.
    labels: array-like or list.
        Corresponding labels for each sample.
    idx: list of ints
        Indices for fold.
    ksplit: int
        Number of folds.
    name: string
        Name of classifier.
    mdl: TODO
    param: TODO
        Parameters for the classifier.
    fld: TODO
    """

    if name not in NAMES:
        raise ValueError("Classifier %s not supported. Did you enter it properly?" % name)

    print 'Tuning %s hyper-parameters' %(name)
    # Redefine the parameters to be used for RBF SVM (dependent on
    # training data)
    if name == 'RBF SVM':
        #Euclidean distances between samples
        dist = pdist(data[fld[0],:], 'euclidean').ravel()
        #Estimates for sigma (10th, 50th and 90th percentile)
        sigest = np.asarray(np.percentile(dist,[10,50,90]))
        #Estimates for gamma (= -1/(2*sigma^2))
        gamma = 1./(2*sigest**2)
        #Set SVM parameters with these values
        param = [{'kernel': ['rbf'],
                  'gamma': gamma.tolist(),
                  'C': np.logspace(-2,2,5).tolist()}]

        #Run Grid Search with parallel processing
    if name == "Decision Tree" or name == "Naive Bayes":
        clf = mdl
    else:
        clf = GridSearchCV(mdl, param, cv=5, scoring='f1', n_jobs=PROCESSORS)

    if True:  #better identifier here
        fScore = []
        for fold in range(ksplit):
            fScore.append(wrapper_clf(fold, data, labels, idx, clf=clf))
    else:
        pool=MemmapingPool(processes=min(ksplit, PROCESSORS))
        fScore = pool.map(functools.partial(wrapper_clf, data, labels, idx, clf=clf),
                          range(ksplit))
        pool.close()
        pool.join()

    return clf, fScore

def wrapper_clf(fold, data, labels, idx, clf=None):
    """
    TODO
    """
    id = idx[fold]
    clf.fit(data[id[0], :], labels[id[0]])
    fpr, tpr, thresholds = rc(labels[id[1]],
                              clf.predict_proba(data[id[1], :])[:, 1])
    return auc(fpr,tpr)

def load_data(source_dir, data_pattern):
    data_files = glob(path.join(source_dir, data_pattern))
    if len(data_files) == 0:
        raise ValueError("No data files found with pattern %s" % data_pattern)

    data = None
    for data_file in data_files:
        d = np.load(data_file)
        if data is not None:
            assert d.shape[0] == data.shape[0]
            data = np.concatenate((data, d), axis=1)
        else:
            data = d

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

    label_files = glob(path.join(source_dir, label_pattern))
    if len(label_files) == 0:
        raise ValueError("No label files found with pattern %s" % data_pattern)
    if len(label_files) > 1:
        raise ValueError("Only one label file supported ATM.")

    return np.load(label_files[0]).flatten()

def main(source_dir, ksplit, out_dir, data_pattern, label_pattern):
    #load activations and labels
    data = load_data(source_dir, data_pattern)
    labels = load_labels(source_dir, label_pattern)

    # Get classifiers and params
    classifiers, params = make_classifiers(data.shape, ksplit)

    # preprocess dataset, split into training and test part
    X, y = data, labels
    X = StandardScaler().fit_transform(X)
    kf = StratifiedKFold(labels,
                         n_folds=ksplit)
    idx = [(tr,ts) for (tr, ts) in kf]

    score={}
    dscore=[]
    for name, mdl, param, fld in zip(NAMES, classifiers, params, idx):
        clf, fScore = get_score(data, labels, idx, ksplit, name,
                           mdl, param, fld)

        if out_dir is not None:
            with open(path.join(out_dir, name + "%.2f.pkl" % (np.mean(fScore))), "wb") as f:
                pickle.dump(clf,f)

        dscore.append(fScore)
        score[name] = (np.mean(fScore), np.std(fScore))

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
    parser.add_argument("--out", default=None, help="Output directory of files.")
    parser.add_argument("--folds", default=10, help="Number of folds for n-fold cross validation")
    parser.add_argument("--data_pattern", default="data.npy", help="Pattern for data files")
    parser.add_argument("--label_pattern", default="labels.npy", help="Pattern for label files")
    return parser

if __name__ == "__main__":
    CPUS = multiprocessing.cpu_count()
    if CPUS < PROCESSORS: raise ValueError("Number of PROCESSORS exceed available CPUs, please edits this in the script and come again!")

    parser = make_argument_parser()
    args = parser.parse_args()
    main(args.data_directory, out_dir=args.out, ksplit=int(args.folds),
         data_pattern=args.data_pattern, label_pattern=args.label_pattern)
