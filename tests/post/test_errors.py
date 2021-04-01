# post/test_errors.py
"""Tests for rom_operator_inference.post._errors.py."""

import pytest
import numpy as np
import scipy.linalg as la

import rom_operator_inference as opinf


def test_absolute_and_relative_error(set_up_error_data):
    """Test post._errors._absolute_and_relative_error() (helper function)."""
    error_data = set_up_error_data
    X, Y = error_data.truth, error_data.approximation

    # Frobenious norm
    abs_err, rel_err = opinf.post._errors._absolute_and_relative_error(X, Y,
                                                                       la.norm)
    assert isinstance(abs_err, float)
    assert isinstance(rel_err, float)

    def eucnorm(z):
        """Euclidean norm, columnwise."""
        return la.norm(z, axis=0, ord=2)

    # Euclidean norm
    abs_err, rel_err = opinf.post._errors._absolute_and_relative_error(X, Y,
                                                                       eucnorm)
    assert abs_err.shape == rel_err.shape == (X.shape[1],)


def test_frobenius_error(set_up_error_data):
    """Test post.frobenius_error()."""
    error_data = set_up_error_data
    X, Y, _ = error_data.truth, error_data.approximation, error_data.time

    # Try with bad shapes.
    with pytest.raises(ValueError) as exc:
        opinf.post.frobenius_error(X, Y[:,:-1])
    assert exc.value.args[0] == "truth X and approximation Y not aligned"

    with pytest.raises(ValueError) as exc:
        opinf.post.frobenius_error(np.dstack((X,X)), np.dstack((Y,Y)))
    assert exc.value.args[0] == "X and Y must be two-dimensional"

    # Test correct usage.
    abs_err, rel_err = opinf.post.frobenius_error(X, Y)
    assert isinstance(abs_err, float)
    assert isinstance(rel_err, float)


def test_lp_error(set_up_error_data):
    """Test post.lp_error()."""
    error_data = set_up_error_data
    X, Y = error_data.truth, error_data.approximation

    # Try with invalid p.
    with pytest.raises(ValueError) as exc:
        opinf.post.lp_error(X, Y, p=-1)
    assert exc.value.args[0] == "norm order p must be positive (np.inf ok)"

    # Try with bad shapes.
    with pytest.raises(ValueError) as exc:
        opinf.post.lp_error(X, Y[:,:-1])
    assert exc.value.args[0] == "truth X and approximation Y not aligned"

    with pytest.raises(ValueError) as exc:
        opinf.post.lp_error(np.dstack((X,X)), np.dstack((Y,Y)))
    assert exc.value.args[0] == "X and Y must be one- or two-dimensional"

    # Test correct usage.
    for p in [1, 2, 5.7, np.inf]:
        abs_err, rel_err1 = opinf.post.lp_error(X, Y, p=p)
        assert abs_err.shape == rel_err1.shape == (X.shape[1],)

        abs_err, rel_err2 = opinf.post.lp_error(X, Y, p=p, normalize=True)
        assert abs_err.shape == rel_err2.shape == (X.shape[1],)
        assert np.all(rel_err1 >= rel_err2)

        abs_err1D, rel_err1D = opinf.post.lp_error(X[:,0], Y[:,0], p=p)
        assert isinstance(abs_err1D, float)
        assert isinstance(rel_err1D, float)


def test_Lp_error(set_up_error_data):
    """Test post.Lp_error()."""
    error_data = set_up_error_data
    X, Y, t = error_data.truth, error_data.approximation, error_data.time

    # Try with invalid p.
    with pytest.raises(ValueError) as exc:
        opinf.post.Lp_error(X, Y, p=-1)
    assert exc.value.args[0] == "norm order p must be positive (np.inf ok)"

    # Try with bad shapes.
    with pytest.raises(ValueError) as exc:
        opinf.post.Lp_error(X, Y[:,:-1], t)
    assert exc.value.args[0] == "truth X and approximation Y not aligned"

    with pytest.raises(ValueError) as exc:
        opinf.post.Lp_error(np.dstack((X,X)), np.dstack((Y,Y)), t)
    assert exc.value.args[0] == "X and Y must be one- or two-dimensional"

    with pytest.raises(ValueError) as exc:
        opinf.post.Lp_error(X, Y, np.dstack((t,t)))
    assert exc.value.args[0] == "time t must be one-dimensional"

    with pytest.raises(ValueError) as exc:
        opinf.post.Lp_error(X, Y, t[:-1])
    assert exc.value.args[0] == "truth X not aligned with time t"

    # Try bad combination of t and p.
    with pytest.raises(ValueError) as exc:
        opinf.post.Lp_error(X, Y, p=2)
    assert exc.value.args[0] == "time t required for p < infinty"

    # Test correct usage.
    for p in [1, 2, 5.7]:
        abs_err, rel_err = opinf.post.Lp_error(X, Y, t, p=p)
        assert isinstance(abs_err, float)
        assert isinstance(rel_err, float)

    abs_err, rel_err = opinf.post.Lp_error(X, Y, p=np.inf)
    assert isinstance(abs_err, float)
    assert isinstance(rel_err, float)

    abs_err2, rel_err2 = opinf.post.Lp_error(X, Y, t, p=np.inf)
    assert isinstance(abs_err2, float)
    assert isinstance(rel_err2, float)
    assert abs_err == abs_err2
    assert rel_err == rel_err2

    # Test 1D inputs.
    for p in [1, 2, 5.7]:
        abs_err, rel_err = opinf.post.Lp_error(X[0], Y[0], t, p=p)
        assert isinstance(abs_err, float)
        assert isinstance(rel_err, float)

    # Do a 1D numerical test.
    t = np.linspace(0, np.pi, 400)
    X = np.sin(t)
    Y = np.sin(2*t)

    abs_err, rel_err = opinf.post.Lp_error(X, Y, t, p=1)
    assert round(abs_err, 4) == 2.5
    assert round(rel_err, 4) == 1.25

    abs_err, rel_err = opinf.post.Lp_error(X, Y, t, p=2)
    assert round(abs_err, 4) == round(np.sqrt(np.pi), 4)
    assert round(rel_err, 4) == round(np.sqrt(2), 4)

    abs_err, rel_err = opinf.post.Lp_error(X, Y, t, p=np.inf)
    assert round(abs_err, 4) == 1.7602
    assert round(rel_err, 4) == 1.7602

    # Do a 2D numerical test.
    t = np.linspace(0, np.pi, 400)
    X = np.vstack([np.sin(t), np.cos(t)])
    Y = np.zeros_like(X)

    abs_err, rel_err = opinf.post.Lp_error(X, Y, t, p=1)
    assert round(abs_err, 4) == 4
    assert rel_err == 1

    abs_err, rel_err = opinf.post.Lp_error(X, Y, t, p=2)
    assert round(abs_err, 4) == round(np.sqrt(np.pi), 4)
    assert rel_err == 1

    abs_err, rel_err = opinf.post.Lp_error(X, Y, p=np.inf)
    assert round(abs_err, 4) == 1
    assert rel_err == 1
