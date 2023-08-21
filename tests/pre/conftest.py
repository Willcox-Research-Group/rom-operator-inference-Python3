# pre/conftest.py
"""Fixtures for testing the pre submodule."""

import pytest
import numpy as np


@pytest.fixture
def set_up_transformer_data():
    n = 2000
    k = 500
    return np.random.random((n, k)) - .5
