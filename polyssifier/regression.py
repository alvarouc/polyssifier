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
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression
import time
from sklearn.preprocessing import LabelEncoder
from itertools import starmap
from sklearn import datasets

#These are commented out for now
#from .poly_utils import build_classifiers, MyVoter
#from .report import Report

sys.setrecursionlimit(10000)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def main():
    print("testing")
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target
    print(X)
    print(Y)
    print(linear_regress(X, Y))
    print(multivariate_regress(X, Y, 5))


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


def linear_regress(independent_variables, predictor):
    '''
    :param independent_variables: the data (numpy matrix) for which we will use to predict the predictor
    :param predictor: the data (vector) which we want to predict
    :return: the mean cross validation score given 10 partitions of the dataset
    '''
    regr = LinearRegression()
    return np.mean(cross_val_score(regr, independent_variables, predictor, cv=10))

def multivariate_regress(independent_variables, predictor, degree):
    '''
    :param independent_variables: the data (numpy matrix) for which we will use to predict the predictor
    :param predictor: the data (vector) which we want to predict
    :param degree: how many instances of each independent_variable type will be replicated
    :return: the mean cross validation score given 10 partitions of the dataset
    '''

    tup = independent_variables.shape
    width = tup[0]
    height = tup[1]
    copy = np.copy(independent_variables)

    for i in range(degree - 1):
        to_concat = np.zeros((width, height))
        to_concat.flags.writeable = True
        for j in range(width):
            for k in range(height):
                to_multiply = 1
                for l in range(i + 1):
                    to_concat = to_multiply * independent_variables[j][k]
                print(j)
                print(k)
                to_concat[j][k] = to_multiply
        np.concatenate(copy, to_concat)

    return linear_regress(copy, predictor, degree)


if __name__ == "__main__":
    main()