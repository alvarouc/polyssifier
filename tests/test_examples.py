import pytest
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore",category=DeprecationWarning)

from sklearn import datasets
from polyssifier import Poly
import numpy as np
import pandas as pd
from sklearn.datasets import make_moons

@pytest.mark.medium
def test_run():
    data, label = make_moons(n_samples=2000, noise=0.4)
    pol = Poly(data,label, n_folds=2, verbose=1, feature_selection=False)
    scores= pol.run()
