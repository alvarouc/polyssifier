import pytest
import warnings

from polyssifier import poly
from sklearn.datasets import make_classification

warnings.filterwarnings("ignore", category=DeprecationWarning)

NSAMPLES = 100
data, label = make_classification(n_samples=NSAMPLES, n_features=50,
                                  n_informative=10, n_redundant=10,
                                  n_repeated=0, n_classes=2,
                                  n_clusters_per_class=2, weights=None,
                                  flip_y=0.01, class_sep=2.0,
                                  hypercube=True, shift=0.0,
                                  scale=1.0, shuffle=True,
                                  random_state=1988)


@pytest.mark.medium
def test_run():
    report = poly(data, label, n_folds=5, verbose=1,
                  feature_selection=False,
                  save=False, project_name='test2')
    assert (report.scores.mean()[:, 'test'] > 0.5).all(),\
        'test score below chance'
    assert (report.scores.mean()[:, 'train'] > 0.5).all(),\
        'train score below chance'


@pytest.mark.medium
def test_feature_selection():
    report = poly(data, label, n_folds=5, verbose=1,
                  feature_selection=True,
                  save=False, project_name='test2')
    assert (report.scores.mean()[:, 'test'] > 0.5).all(),\
        'test score below chance'
    assert (report.scores.mean()[:, 'train'] > 0.5).all(),\
        'train score below chance'


@pytest.mark.medium
def test_plot():
    report = poly(data, label, n_folds=5, verbose=1,
                  feature_selection=True,
                  save=False, project_name='test2')
    report.plot_scores()
