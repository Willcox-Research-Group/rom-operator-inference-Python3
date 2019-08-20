# test_pre.py
"""Tests for rom_operator_inference.pre.py"""

import pytest
import numpy as np
from scipy import linalg as la
from collections import namedtuple

import rom_operator_inference as roi


ErrorData = namedtuple("ErrorData", ["truth", "approximation", "time"])

@pytest.fixture
def set_up_error_data():
    n = 200
    k = 50
    X = np.random.random((n,k)) - .5
    Y = X + np.random.normal(loc=0, scale=1e-4, size=(n,k))
    t = np.linspace(0, 1, k)
    return ErrorData(X, Y, t)


def test_discrete_error(set_up_error_data):
    """Test post.discrete_error()."""
    error_data = set_up_error_data
    X, Y = error_data.truth, error_data.approximation

    abs_err, rel_err = roi.post.discrete_error(X, Y)
    assert abs_err.shape == rel_err.shape == (X.shape[1],)

    abs_err1D, rel_err1D = roi.post.discrete_error(X[:,0], Y[:,0])
    assert isinstance(abs_err1D, float)
    assert isinstance(rel_err1D, float)

    # Try with bad shapes.
    with pytest.raises(ValueError) as exc:
        roi.post.discrete_error(X, Y[:,:-1])
    assert exc.value.args[0] == "truth X and approximation Y not aligned"

    with pytest.raises(ValueError) as exc:
        roi.post.discrete_error(np.dstack((X,X)), np.dstack((Y,Y)))
    assert exc.value.args[0] == "X and Y must be one- or two-dimensional"


def test_continuous_error(set_up_error_data):
    """Test post.continuous_error()."""
    error_data = set_up_error_data
    X, Y, t = error_data.truth, error_data.approximation, error_data.time

    abs_err, rel_err = roi.post.continuous_error(X, Y, t)
    assert isinstance(abs_err, float)
    assert isinstance(rel_err, float)

    # Try with bad shapes.
    with pytest.raises(ValueError) as exc:
        roi.post.continuous_error(X, Y[:,:-1], t)
    assert exc.value.args[0] == "truth X and approximation Y not aligned"

    with pytest.raises(ValueError) as exc:
        roi.post.continuous_error(np.dstack((X,X)), np.dstack((Y,Y)), t)
    assert exc.value.args[0] == "X and Y must be two-dimensional"

    with pytest.raises(ValueError) as exc:
        roi.post.continuous_error(X, Y, np.dstack((t,t)))
    assert exc.value.args[0] == "time t must be one-dimensional"

    with pytest.raises(ValueError) as exc:
        roi.post.continuous_error(X, Y, t[:-1])
    assert exc.value.args[0] == "truth X not aligned with time t"
