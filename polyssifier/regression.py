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
from sklearn.linear_model import LinearRegression, BayesianRidge, Perceptron
from sklearn.gaussian_process import GaussianProcessRegressor
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

#Currently, this main method is used for testing
def main():
    print("testing")
    #This sets the returning arary to not print in scientific notation
    np.set_printoptions(suppress=True)
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target
    print(linear_regress(X, Y))
    print(multivariate_regress(X, Y, 3))
    print(gaussian_proccess_regression(X, Y))
    print(bayesian_ridge_regression(X, Y))
    print(perceptron_regression(X, Y))

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

    #First we make an empty matrix which is the size of what we wish to pass through to linear regress
    height_of_pass_through = independent_variables.shape[0]
    width_of_pass_through = degree * independent_variables.shape[1]
    to_pass_through = np.zeros(shape=(height_of_pass_through, width_of_pass_through))

    #These are the width and height of each "exponeneted" matrix
    height_exponential_matrix = independent_variables.shape[0]
    width_exponential_matrix = independent_variables.shape[1]

    for i in range(degree):
        to_add_in = exponent_matrix(independent_variables, (i + 1))
        for j in range(height_exponential_matrix):
            for k in range(width_exponential_matrix):
                to_pass_through.itemset((j, k + i * width_exponential_matrix), (to_add_in.item(j, k)))
    return linear_regress(to_pass_through, predictor)

def exponent_matrix(matrix, exponent):
    '''
    :param matrix: the numpy matrix which will be scaled
    :param exponent: the exponent which each individual entry in the matrix will be raised to
    :return: a new matrix (the old one is not modified) where each value is the old matrix's value raised
    to the exponent
    '''
    to_return = np.copy(matrix)
    for i in range(to_return.shape[0]):
        for j in range(to_return.shape[1]):
            to_return.itemset((i,j), ((matrix.item(i,j)**exponent)))
    return to_return

def gaussian_proccess_regression(independent_variables, predictor):

    #The kernel used by default in this Gaussian Regression is the radial basis function kernel
    regr = GaussianProcessRegressor()
    return np.mean(cross_val_score(regr, independent_variables, predictor, cv=10))

def bayesian_ridge_regression(independent_variables, predictor):
    regr = BayesianRidge()
    return np.mean(cross_val_score(regr, independent_variables, predictor, cv=10))

def perceptron_regression(independent_variables, predictor):
    regr = Perceptron()
    return np.mean(cross_val_score(regr, independent_variables, predictor, cv=10))

if __name__ == "__main__":
    main()