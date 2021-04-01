# pre/conftest.py
"""Fixtures for testing the pre submodule."""

import pytest
import numpy as np
from collections import namedtuple


@pytest.fixture
def set_up_basis_data():
    n = 2000
    k = 500
    return np.random.random((n,k)) - .5


DynamicState = namedtuple("DynamicState", ["time", "state", "derivative"])


def _difference_data(t):
    Y = np.row_stack((t,
                      t**2/2,
                      t**3/3,
                      np.sin(t),
                      np.exp(t),
                      1/(t+1),
                      t + t**2/2 + t**3/3 + np.sin(t) - np.exp(t)
                      ))
    dY = np.row_stack((np.ones_like(t),
                       t,
                       t**2,
                       np.cos(t),
                       np.exp(t),
                       -1/(t+1)**2,
                       1 + t + t**2 + np.cos(t) - np.exp(t)
                       ))
    return DynamicState(t, Y, dY)


@pytest.fixture
def set_up_uniform_difference_data():
    t = np.linspace(0, 1, 400)
    return _difference_data(t)


@pytest.fixture
def set_up_nonuniform_difference_data():
    t = np.linspace(0, 1, 400)**2
    return _difference_data(t)
