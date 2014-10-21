#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
This module computes the baseline results by applying various
ML classifiers such as SVM, LDA, Naive bayes, K-NN.
"""

import os, sys
import numpy as np
from scipy.spatial.distance import pdist
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as pl
from multiprocessing import Pool,Array,Process,Manager
import functools
import seaborn as sb

from sklearn.metrics import auc
from sklearn.metrics import roc_curve as rc
from sklearn.grid_search import GridSearchCV

from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier

from scipy.io import savemat
import random as rndc

# Number of folds for n-fold cross validation
ksplit=10
DATADIR='./' # directory where your labels and the data are
PROCESSORS=75 # please set this number to no more than the number of cores on the machine you're going to be running it on but high enough to help the computation

fdata = 'data.npy'
flabels= 'labels.npy'

#load activations and labels
data=np.load(DATADIR + fdata)
labels=np.load(DATADIR+flabels).flatten()

seed = rndc.SystemRandom().seed()

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",  "Decision Tree",
         "Random Forest", "Logistic Regression", "Naive Bayes", "LDA"]
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
    [{'n_neighbors': range(1,20)}],                             # KNN
    [{'kernel': ['linear'],'C': [1]}], # linearSVM
    [{'kernel': ['rbf'], 
      'gamma': np.arange(0.1,1,0.1).tolist()+range(1,10),
      'C': np.logspace(-2,2,5).tolist()}],                      # rbf SVM
    [],                                                         # decision tree
    [{'n_estimators': range(5,20)}],                            # random forest
    [{'C': np.logspace(0.1,3,7).tolist()}],                     # logistic regression
    [],                                                         # Naive Bayes
    [{'n_components': [np.int(0.1*data.shape[0]),
                       np.int(0.2*data.shape[0]),
                       np.int(0.3*data.shape[0]),
                       np.int(0.5*data.shape[0]),
                       np.int(0.7*data.shape[0])]}],             # LDA                 
]
# preprocess dataset, split into training and test part
X, y = data, labels
X = StandardScaler().fit_transform(X)
kf = StratifiedKFold(labels,n_folds=ksplit)
idx = [(tr,ts) for tr,ts in kf]

# iterate over classifiers
def wrapper_clf(fold, clf=None):
    id = idx[fold]
    clf.fit(data[id[0],:], labels[id[0]])
    fpr, tpr, thresholds = rc(labels[id[1]], 
                              clf.predict_proba(data[id[1],:])[:, 1])
    return auc(fpr,tpr)

score={}
dscore=[]
for name, mdl, param, fld in zip(names, classifiers, params, idx):
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
	clf=mdl
    else:
    	clf = GridSearchCV(mdl, param, cv=5, scoring='f1', n_jobs=PROCESSORS)
    if True:    
        fScore = []
        for fold in range(ksplit):
            fScore.append(wrapper_clf(fold, clf=clf))
    else:
        pool=Pool(processes=min(ksplit, PROCESSORS))
        fScore = pool.map(functools.partial(wrapper_clf, clf=clf), 
                          range(ksplit))
        pool.close()
        pool.join()    
    dscore.append(fScore)
    score[name]=(np.mean(fScore),np.std(fScore))

dscore = np.asarray(dscore)

#plot bar charts for the various classifiers
pl.figure(figsize=[10,6])
ax=pl.gca()
sb.barplot(np.array(names), dscore, palette="Paired")
ax.set_xticks(np.arange(len(names)))
ax.set_xticklabels(names, rotation=30)
ax.set_ylabel('classification AUC')
pl.subplots_adjust(bottom=0.18)
#pl.draw()
# change the file you're saving it to
pl.savefig('polyssifier.png')
#pl.show(True)
