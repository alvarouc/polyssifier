import pytest
import numpy as np
import sys
sys.path.append('../')
from polyssifier.polyssifier import create_polynomial


@pytest.mark.medium
def test_create_polynomial():
    data = np.array([[5, 6], [7, 8]])
    poly1 = np.array([[5, 6], [7, 8]])
    poly2 = np.array([[5, 6, 25, 36], [7, 8, 49, 64]])
    poly3 = np.array([[5, 6, 25, 36, 125, 216], [7, 8, 49, 64, 343, 512]])
    assert (poly1 == create_polynomial(data, 1)).all()
    assert (poly2 == create_polynomial(data, 2)).all()
    assert (poly3 == create_polynomial(data, 3)).all()
