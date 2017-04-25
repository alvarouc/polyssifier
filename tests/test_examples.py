import pytest
import warnings
from polyssifier import poly, plot
from sklearn.datasets import make_moons, make_classification

warnings.filterwarnings("ignore", category=DeprecationWarning)

NSAMPLES = 1000

@pytest.mark.medium
def test_run():
    data, label = make_classification(n_samples=NSAMPLES, n_features=100,
                                      n_informative=50, n_redundant=50,
                                      n_repeated=0, n_classes=2,
                                      n_clusters_per_class=2, weights=None,
                                      flip_y=0.01, class_sep=1.0,
                                      hypercube=True, shift=0.0,
                                      scale=1.0, shuffle=True,
                                      random_state=1988)
    scores, confusions, predictions, test_proba = \
        poly(data, label, n_folds=5, verbose=1, feature_selection=False,
             exclude=['Multilayer Perceptron'],
             save=False, project_name='test2')

    # scores, confusions, predictions, test_proba = \
    #     poly(data, label, n_folds=3, verbose=1,
    #          exclude=['Multilayer Perceptron'], feature_selection=True,
    #          project_name='test3')
    # scores, confusions, predictions, test_proba = \
    #     poly(data, label, n_folds=3, verbose=1,
    #          exclude=['Multilayer Perceptron',
    #                   'Voting'],
    #          feature_selection=True,
    #          project_name='test3')
    # plot(scores)
