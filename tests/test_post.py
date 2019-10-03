# test_pre.py
"""Tests for rom_operator_inference.post.py"""

import pytest
import numpy as np
from scipy import linalg as la
from collections import namedtuple

import rom_operator_inference as roi


ErrorData = namedtuple("ErrorData", ["truth", "approximation", "time"])

@pytest.fixture
def set_up_error_data():
    n = 2000
    k = 500
    X = np.random.random((n,k)) - .5
    Y = X + np.random.normal(loc=0, scale=1e-4, size=(n,k))
    t = np.linspace(0, 1, k)
    return ErrorData(X, Y, t)


def test_absolute_and_relative_error(set_up_error_data):
    """Test post._absolute_and_relative_error() (helper function)."""
    error_data = set_up_error_data
    X, Y = error_data.truth, error_data.approximation

    # Frobenious norm
    abs_err, rel_err = roi.post._absolute_and_relative_error(X, Y, la.norm)
    assert isinstance(abs_err, float)
    assert isinstance(rel_err, float)

    # Euclidean norm, columnwise
    eucnorm = lambda z: la.norm(z, axis=0)
    abs_err, rel_err = roi.post._absolute_and_relative_error(X, Y, eucnorm)
    assert abs_err.shape == rel_err.shape == (X.shape[1],)


def test_frobenius_error(set_up_error_data):
    """Test post.frobenius_error()."""
    error_data = set_up_error_data
    X, Y, t = error_data.truth, error_data.approximation, error_data.time

    # Try with bad shapes.
    with pytest.raises(ValueError) as exc:
        roi.post.frobenius_error(X, Y[:,:-1])
    assert exc.value.args[0] == "truth X and approximation Y not aligned"

    with pytest.raises(ValueError) as exc:
        roi.post.frobenius_error(np.dstack((X,X)), np.dstack((Y,Y)))
    assert exc.value.args[0] == "X and Y must be two-dimensional"

    # Test correct usage.
    abs_err, rel_err = roi.post.frobenius_error(X, Y)
    assert isinstance(abs_err, float)
    assert isinstance(rel_err, float)



def test_lp_error(set_up_error_data):
    """Test post.lp_error()."""
    error_data = set_up_error_data
    X, Y = error_data.truth, error_data.approximation

    # Try with invalid p.
    with pytest.raises(ValueError) as exc:
        roi.post.lp_error(X, Y, p=-1)
    assert exc.value.args[0] == "norm order p must be positive (np.inf ok)"

    # Try with bad shapes.
    with pytest.raises(ValueError) as exc:
        roi.post.lp_error(X, Y[:,:-1])
    assert exc.value.args[0] == "truth X and approximation Y not aligned"

    with pytest.raises(ValueError) as exc:
        roi.post.lp_error(np.dstack((X,X)), np.dstack((Y,Y)))
    assert exc.value.args[0] == "X and Y must be one- or two-dimensional"

    # Test correct usage.
    for p in [1, 2, 5.7, np.inf]:
        abs_err, rel_err1 = roi.post.lp_error(X, Y, p=p)
        assert abs_err.shape == rel_err1.shape == (X.shape[1],)

        abs_err, rel_err2 = roi.post.lp_error(X, Y, p=p, normalize=True)
        assert abs_err.shape == rel_err2.shape == (X.shape[1],)
        assert np.all(rel_err1 >= rel_err2)

        abs_err1D, rel_err1D = roi.post.lp_error(X[:,0], Y[:,0], p=p)
        assert isinstance(abs_err1D, float)
        assert isinstance(rel_err1D, float)


def test_Lp_error(set_up_error_data):
    """Test post.Lp_error()."""
    error_data = set_up_error_data
    X, Y, t = error_data.truth, error_data.approximation, error_data.time

    # Try with invalid p.
    with pytest.raises(ValueError) as exc:
        roi.post.Lp_error(X, Y, p=-1)
    assert exc.value.args[0] == "norm order p must be positive (np.inf ok)"

    # Try with bad shapes.
    with pytest.raises(ValueError) as exc:
        roi.post.Lp_error(X, Y[:,:-1], t)
    assert exc.value.args[0] == "truth X and approximation Y not aligned"

    with pytest.raises(ValueError) as exc:
        roi.post.Lp_error(np.dstack((X,X)), np.dstack((Y,Y)), t)
    assert exc.value.args[0] == "X and Y must be two-dimensional"

    with pytest.raises(ValueError) as exc:
        roi.post.Lp_error(X, Y, np.dstack((t,t)))
    assert exc.value.args[0] == "time t must be one-dimensional"

    with pytest.raises(ValueError) as exc:
        roi.post.Lp_error(X, Y, t[:-1])
    assert exc.value.args[0] == "truth X not aligned with time t"

    # Try bad combination of t and p.
    with pytest.raises(ValueError) as exc:
        roi.post.Lp_error(X, Y, p=2)
    assert exc.value.args[0] == "time t required for p < infinty"

    # Test correct usage.
    for p in [1, 2, 5.7]:
        abs_err, rel_err = roi.post.Lp_error(X, Y, t, p=p)
        assert isinstance(abs_err, float)
        assert isinstance(rel_err, float)

    abs_err, rel_err = roi.post.Lp_error(X, Y, p=np.inf)
    assert isinstance(abs_err, float)
    assert isinstance(rel_err, float)

    abs_err, rel_err = roi.post.Lp_error(X, Y, t, p=np.inf)
    assert isinstance(abs_err, float)
    assert isinstance(rel_err, float)
