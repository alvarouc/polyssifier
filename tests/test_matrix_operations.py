import pytest
import warnings
import numpy as np
from polyssifier import exponent_matrix, create_multivariate

warnings.filterwarnings("ignore", category=DeprecationWarning)

@pytest.mark.medium
def test_exponent_matrix():
    matrix = np.matrix('1 2; 3 4')
    matrix_2pow = np.matrix('1 4; 9 16')
    matrix_3pow = np.matrix('1 8; 27 64')
    matrix_4pow = np.matrix('1 16; 81 256')
    assert matrix == exponent_matrix(matrix, 1)
    assert matrix_2pow == exponent_matrix(matrix, 2)
    assert matrix_3pow == exponent_matrix(matrix, 3)
    assert matrix_4pow == exponent_matrix(matrix, 4)