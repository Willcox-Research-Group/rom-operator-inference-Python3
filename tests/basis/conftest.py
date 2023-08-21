# basis/conftest.py
"""Fixtures for testing the basis submodule."""

import pytest
import numpy as np


@pytest.fixture
def set_up_basis_data():
    n = 2000
    k = 500
    return np.random.random((n, k)) - .5
