# post/conftest.py
"""Fixtures for testing the post submodule."""

import pytest
import numpy as np
from collections import namedtuple


ErrorData = namedtuple("ErrorData", ["truth", "approximation", "time"])


@pytest.fixture
def set_up_error_data():
    n = 2000
    k = 500
    X = np.random.random((n,k)) - .5
    Y = X + np.random.normal(loc=0, scale=1e-4, size=(n,k))
    t = np.linspace(0, 1, k)
    return ErrorData(X, Y, t)
