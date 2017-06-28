import pytest
import warnings
import numpy as np
from polyssifier import polyr
from sklearn.datasets import load_diabetes, load_boston

warnings.filterwarnings("ignore", category=DeprecationWarning)
diabetes_data = load_diabetes().data
diabetes_target = load_diabetes().target
boston_data = load_boston().data
boston_target = load_boston().target


@pytest.mark.medium
def test_feature_selection_regression():
    global report_with_features
    report_with_features = polyr(diabetes_data, diabetes_target, n_folds=2,
                                 verbose=1, concurrency=1,
                                 feature_selection=True, scoring='r2',
                                 save=False, project_name='test_feature_selection')
    assert (report_with_features.scores.mean()[:, 'test'] > 0.2).all(),\
        'test score below chance'
    assert (report_with_features.scores.mean()[:, 'train'] > 0.2).all(),\
        'train score below chance'

    for key, ypred in report_with_features.predictions.iteritems():
        mse = np.linalg.norm(ypred - diabetes_target) / len(diabetes_target)
        assert mse < 5, '{} Prediction error is too high'.format(key)


@pytest.mark.medium
def test_run_regression_diabetes():
    global report
    report = polyr(diabetes_data, diabetes_target, n_folds=2,
                   verbose=1, concurrency=1,
                   feature_selection=False, scoring='r2',
                   save=False, project_name='test_diabetes')
    assert (report.scores.mean()[:, 'test'] > 0.2).all(),\
        'test score below chance'
    assert (report.scores.mean()[:, 'train'] > 0.2).all(),\
        'train score below chance'

@pytest.mark.medium
def test_run_regression_boston():
    global report
    report = polyr(boston_data, boston_target, n_folds=2,
                   verbose=1, concurrency=1,
                   feature_selection=False, scoring='r2',
                   save=False, project_name='test_boston')
    assert (report.scores.mean()[:, 'test'] > 0.2).all(),\
        'test score below chance'
    assert (report.scores.mean()[:, 'train'] > 0.2).all(),\
        'train score below chance'

@pytest.mark.medium
def test_polynomial_model_diabetes():
    #Lars excluded as it performs poorly.
    polynomial_report = polyr(diabetes_data, diabetes_target, n_folds=2, num_degrees=3,
                              verbose=1, concurrency=1, feature_selection=False, save=False,
                              project_name='polynomial_test_diabetes', exclude=['Lars'])
    assert (polynomial_report.scores.mean()[:, 'test'] > 0.3).all(), \
        'test score below chance'

@pytest.mark.medium
def test_polynomial_model_boston():
    polynomial_report = polyr(boston_data, boston_target, n_folds=10, num_degrees=3,
                              verbose=1, concurrency=1, feature_selection=False, save=False,
                              project_name='polynomial_test_boston')
    assert (polynomial_report.scores.mean()[:, 'test'] > 0.3).all(), \
        'test score below chance'


@pytest.mark.medium
def test_plot_scores_no_selection():
    report.plot_scores()
    report.plot_features()


@pytest.mark.medium
def test_plot_features_with_selection():
    report_with_features.plot_scores()
    report_with_features.plot_features()