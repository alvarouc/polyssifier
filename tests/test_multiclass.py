import matplotlib  # noqa: E402
matplotlib.use('Agg')  # noqa: E402
import sys
sys.path.append('../')  # noqa: E402
from polyssifier import poly  # noqa: E402

from sklearn.datasets import make_classification
import warnings
import pytest

warnings.filterwarnings("ignore", category=DeprecationWarning)

NSAMPLES = 100
N_CLASSES = 5
data, label = make_classification(n_samples=NSAMPLES, n_features=50,
                                  n_informative=10, n_redundant=10,
                                  n_repeated=0, n_classes=N_CLASSES,
                                  n_clusters_per_class=2, weights=None,
                                  flip_y=0.01, class_sep=2.0,
                                  hypercube=True, shift=0.0,
                                  scale=1.0, shuffle=True,
                                  random_state=1988)


def test_run():
    report = poly(data, label, n_folds=2, verbose=1,
                  feature_selection=False,
                  save=False, project_name='test2')
    for key, score in report.scores.mean().iteritems():
        assert score < 5, '{} score is too low'.format(key)


def test_feature_selection():
    global report_with_features
    report_with_features = poly(data, label, n_folds=2, verbose=1,
                                feature_selection=True,
                                save=False, project_name='test2')
    assert (report_with_features.scores.mean()[:, 'test'] > 1/N_CLASSES).all(),\
        'test score below chance'
    assert (report_with_features.scores.mean()[:, 'train'] > 1/N_CLASSES).all(),\
        'train score below chance'


def test_plot_no_selection():
    report = poly(data, label, n_folds=2, verbose=1,
                  feature_selection=False,
                  save=False, project_name='test2')
    report.plot_scores()
    report.plot_features()


# @pytest.mark.medium
# def test_plot_with_selection():
#     report = poly(data, label, n_folds=2, verbose=1,
#                   feature_selection=False,
#                   save=False, project_name='test2')

#     report_with_features.plot_scores()
#     report_with_features.plot_features()
