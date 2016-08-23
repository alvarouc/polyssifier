import pytest
import warnings

from polyssifier import poly, plot
from sklearn.datasets import make_moons, make_classification

warnings.filterwarnings("ignore", category=DeprecationWarning)


@pytest.mark.medium
def test_run():
    data, label = make_moons(n_samples=1000, noise=0.4)
    scores, confusions, predictions = poly(data, label, n_folds=2, verbose=1,
                                           feature_selection=False, save=False,
                                           project_name='test1')
    data, label = make_classification(n_samples=1000, n_features=20,
                                      n_informative=5, n_redundant=2,
                                      n_repeated=0, n_classes=2,
                                      n_clusters_per_class=2, weights=None,
                                      flip_y=0.01, class_sep=1.0,
                                      hypercube=True, shift=0.0,
                                      scale=1.0, shuffle=True,
                                      random_state=None)
    scores, confusions, predictions = poly(data, label, n_folds=3, verbose=1,
                                           feature_selection=False, save=False,
                                           project_name='test2')

    scores, confusions, predictions = poly(data, label, n_folds=3, verbose=1,
                                           exclude=['Multilayer Perceptron',
                                                    'Voting'],
                                           feature_selection=True,
                                           project_name='test3')
    scores, confusions, predictions = poly(data, label, n_folds=3, verbose=1,
                                           exclude=['Multilayer Perceptron',
                                                    'Voting'],
                                           feature_selection=True,
                                           project_name='test3')
    plot(scores)
