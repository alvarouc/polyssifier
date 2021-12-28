import pytest
import warnings
import numpy as np
import os
# import sys
# sys.path.append('../')
from polyssifier import polyr
from sklearn.datasets import load_diabetes

warnings.filterwarnings("ignore")
diabetes_data = load_diabetes().data
diabetes_target = load_diabetes().target


@pytest.mark.medium
def test_feature_selection_regression():
    global report_with_features
    report_with_features = polyr(
        diabetes_data, diabetes_target, n_folds=2,
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
def test_run_regression():
    global report
    report = polyr(diabetes_data, diabetes_target, n_folds=2,
                   verbose=1, concurrency=1,
                   feature_selection=False, scoring='r2',
                   save=False, project_name='test_regression')
    assert (report.scores.mean()[:, 'test'] > 0.2).all(),\
        'test score below chance'
    assert (report.scores.mean()[:, 'train'] > 0.2).all(),\
        'train score below chance'


@pytest.mark.medium
def test_polynomial_model():
    # Lars excluded as it performs poorly.
    polynomial_report = polyr(
        diabetes_data, diabetes_target, n_folds=2, num_degrees=3,
        verbose=1, concurrency=1, feature_selection=False, save=False,
        project_name='polynomial_test', exclude=['Lars'])
    assert (polynomial_report.scores.mean()[:, 'test'] > 0.25).all(), \
        'low test score'


@pytest.mark.medium
def test_plot_scores_no_selection():
    report.plot_scores()
    report.plot_features()


@pytest.mark.medium
def test_plot_features_with_selection():
    report_with_features.plot_scores()
    report_with_features.plot_features()


def setup_function(function):
    """ setup any state tied to the execution of the given function.
    Invoked for every test function in the module.
    """


def teardown_function(function):
    """ teardown any state that was previously setup with a setup_function
    call.
    """

    file_paths = [
        'temp_Bayesian Ridge_feature_ranking.png',
        'temp_Decision Tree_feature_ranking.png',
        'temp_ElasticNet_feature_ranking.png',
        'temp_Lars_feature_ranking.png',
        'temp_Lasso_feature_ranking.png',
        'temp_LassoLars_feature_ranking.png',
        'temp_Linear Regression_feature_ranking.png',
        'temp_Linear SVM_feature_ranking.png',
        'temp_Logistic Regression_feature_ranking.png',
        'temp_OrthogonalMatchingPursuit_feature_ranking.png',
        'temp_PassiveAggressiveRegressor_feature_ranking.png',
        'temp.pdf',
        'temp_Random Forest_feature_ranking.png',
        'temp_Ridge_feature_ranking.png',
        'temp.svg',
        'Report.log',
    ]
    for path in file_paths:
        if os.path.exists(path):
            os.remove(path)
