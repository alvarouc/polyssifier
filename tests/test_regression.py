import pytest
import warnings
import numpy as np
from polyssifier import polyr
from sklearn.datasets import load_diabetes

warnings.filterwarnings("ignore", category=DeprecationWarning)
diabetes_data = load_diabetes().data
diabetes_target = load_diabetes().target


@pytest.mark.medium
def test_feature_selection_regression():
    global report_with_features
    report_with_features = polyr(diabetes_data, diabetes_target, n_folds=2,
                                 verbose=1, concurrency=1,
                                 feature_selection=True, scoring='r2',
                                 save=False, project_name='test3')
    assert (report_with_features.scores.mean()[:, 'test'] > 0.2).all(),\
        'test score below chance'
    assert (report_with_features.scores.mean()[:, 'train'] > 0.2).all(),\
        'train score below chance'

    for key, ypred in report_with_features.predictions.iteritems():
        mse = np.linalg.norm(ypred - diabetes_target) / len(diabetes_target)
        assert mse < 5, '{} Prediction error is too high'.format(key)


@pytest.mark.medium
def test_run_regression():
    global report
    report = polyr(diabetes_data, diabetes_target, n_folds=2,
                   verbose=1, concurrency=1,
                   feature_selection=False, scoring='r2',
                   save=False, project_name='test3')
    assert (report.scores.mean()[:, 'test'] > 0.2).all(),\
        'test score below chance'
    assert (report.scores.mean()[:, 'train'] > 0.2).all(),\
        'train score below chance'

@pytest.mark.medium
def test_polynomial_model():
    #GaussianProcess excluded to increase speed
    polynomial_report = polyr(diabetes_data, diabetes_target, n_folds=2, num_degrees=3,
                              verbose=1, concurrency=1, feature_selection=False, save=False,
                              project_name='polynomial_test', exclude=['GaussianProcessRegressor'])
    assert (polynomial_report.scores.median()[:, 'test'] > 0.2).all(), \
        'test score below chance'


@pytest.mark.medium
def test_plot_scores_no_selection():
    report.plot_scores()
    report.plot_features()


@pytest.mark.medium
def test_plot_features_with_selection():
    report_with_features.plot_scores()
    report_with_features.plot_features()
