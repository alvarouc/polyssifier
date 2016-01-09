import pytest
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore",category=DeprecationWarning)

from polyssifier import Poly
import numpy as np
import pandas as pd
from sklearn.datasets import make_moons, make_classification

@pytest.mark.medium
def test_run():
    data, label = make_moons(n_samples=2000, noise=0.4)
    pol = Poly(data,label, n_folds=2, verbose=1,
               feature_selection=False, save=False)
    scores= pol.run()
    data, label = make_classification(n_samples=2000, n_features=20,
                                      n_informative=5, n_redundant=2,
                                      n_repeated=0, n_classes=5,
                                      n_clusters_per_class=2, weights=None,
                                      flip_y=0.01, class_sep=1.0,
                                      hypercube=True, shift=0.0,
                                      scale=1.0, shuffle=True,
                                      random_state=None)
    pol = Poly(data, label, n_folds=3, verbose=1,
               feature_selection=False, save=False)
    scores= pol.run()

    pol = Poly(data, label, n_folds=3, verbose=1,
               exclude=['Multilayer Perceptron','Voting'],
               feature_selection=True)
    scores = pol.run()
    pol.plot()
#    scores= pol.run()

    
