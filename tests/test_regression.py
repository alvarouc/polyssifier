import pytest
import warnings

from polyssifier import polyr
from sklearn.datasets import load_diabetes

warnings.filterwarnings("ignore", category=DeprecationWarning)
diabetes_data = load_diabetes().data
diabetes_target = load_diabetes().target


@pytest.mark.medium
def test_run_regression():
    report = polyr(diabetes_data, diabetes_target, n_folds=3, verbose=1,
                   feature_selection=False, scoring='r2',
                   save=False, project_name='test3')
    assert (report.scores.mean()[:, 'test'] > 0.2).all(),\
        'test score below chance'
    assert (report.scores.mean()[:, 'train'] > 0.2).all(),\
        'train score below chance'


@pytest.mark.medium
def test_feature_selection_regression():
    report = polyr(diabetes_data, diabetes_target, n_folds=3, verbose=1,
                   feature_selection=True, scoring='r2',
                   save=False, project_name='test3')
    assert (report.scores.mean()[:, 'test'] > 0.2).all(),\
        'test score below chance'
    assert (report.scores.mean()[:, 'train'] > 0.2).all(),\
        'train score below chance'


@pytest.mark.medium
def test_plot_regression():
    report = polyr(diabetes_data, diabetes_target, n_folds=3, verbose=1,
                   feature_selection=False, scoring='r2',
                   save=False, project_name='test3')
    report.plot_scores()
