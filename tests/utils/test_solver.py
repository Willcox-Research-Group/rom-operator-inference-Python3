# utils/test_solver.py
"""Tests for rom_operator_inference.utils._solver.py."""

import pytest
import numpy as np
from scipy import linalg as la

import rom_operator_inference as roi


def test_get_least_squares_size():
    """Test utils._solver.get_least_squares_size()."""
    m, r = 3, 7

    # Try with bad input combinations.
    with pytest.raises(ValueError) as ex:
        roi.utils.get_least_squares_size("cAHB", r)
    assert ex.value.args[0] == "argument m > 0 required since 'B' in modelform"

    with pytest.raises(ValueError) as ex:
        roi.utils.get_least_squares_size("cAH", r, m=10)
    assert ex.value.args[0] == "argument m=10 invalid since 'B' in modelform"

    # Test without inputs.
    assert roi.utils.get_least_squares_size("c", r) == 1
    assert roi.utils.get_least_squares_size("A", r) == r
    assert roi.utils.get_least_squares_size("cA", r) == 1 + r
    assert roi.utils.get_least_squares_size("cAH", r) == 1 + r + r*(r+1)//2
    assert roi.utils.get_least_squares_size("cG", r) == 1 + r*(r+1)*(r+2)//6

    # Test with inputs.
    assert roi.utils.get_least_squares_size("cB", r, m) == 1 + m
    assert roi.utils.get_least_squares_size("AB", r, m) == r + m
    assert roi.utils.get_least_squares_size("cAB", r, m) == 1 + r + m
    assert roi.utils.get_least_squares_size("AHB", r, m) == r + r*(r+1)//2 + m
    assert roi.utils.get_least_squares_size("GB", r, m) == r*(r+1)*(r+2)//6 + m

    # Test with affines.
    assert roi.utils.get_least_squares_size("c", r, affines={"c":[0,0]}) == 2
    assert roi.utils.get_least_squares_size("A", r, affines={"A":[0,0]}) == 2*r


# utils.lstsq_reg() -----------------------------------------------------------
def _test_lstsq_reg_single(k,d,r):
    """Do one test of utils._solver.lstsq_reg()."""
    A = np.random.random((k, d))
    b = np.random.random(k)
    B = np.random.random((k,r))
    I = np.eye(d)

    # VECTOR TEST
    x = la.lstsq(A, b)[0]

    # Ensure that the least squares solution is the usual one when P=0.
    x_ = roi.utils.lstsq_reg(A, b, P=0)[0]
    assert np.allclose(x, x_)

    # Ensure that the least squares solution is the usual one when P=0 (matrix)
    x_ = roi.utils.lstsq_reg(A, b, np.zeros((d,d)))[0]
    assert np.allclose(x, x_)

    # Check that the regularized least squares solution has decreased norm.
    x_ = roi.utils.lstsq_reg(A, b, P=2)[0]
    assert la.norm(x_) <= la.norm(x)

    x_ = roi.utils.lstsq_reg(A, b, P=2*I)[0]
    assert la.norm(x_) <= la.norm(x)

    # MATRIX TEST
    X = la.lstsq(A, B)[0]

    # Ensure that the least squares solution is the usual one when P=0.
    X_ = roi.utils.lstsq_reg(A, B, P=0)[0]
    assert np.allclose(X, X_)

    # Ensure that the least squares solution is the usual one when P=0 (matrix)
    X_ = roi.utils.lstsq_reg(A, B, P=0)[0]
    assert np.allclose(X, X_)

    # Ensure that the least squares solution is the usual one when P=0 (list)
    X_ = roi.utils.lstsq_reg(A, B, P=[np.zeros((d,d))]*r)[0]
    assert np.allclose(X, X_)

    # Ensure that the least squares problem decouples correctly.
    Ps = [l*I for l in range(r)]
    X_ = roi.utils.lstsq_reg(A, B, P=Ps)[0]
    for j in range(r):
        xj_ = roi.utils.lstsq_reg(A, B[:,j], P=Ps[j])[0]
        assert np.allclose(xj_, X_[:,j])

    # Check that the regularized least squares solution has decreased norm.
    X_ = roi.utils.lstsq_reg(A, B, P=2)[0]
    assert la.norm(X_) <= la.norm(X)

    X_ = roi.utils.lstsq_reg(A, B, P=2)[0]
    assert la.norm(X_) <= la.norm(X)

    X_ = roi.utils.lstsq_reg(A, B, P=[2]*r)[0]
    assert la.norm(X_) <= la.norm(X)

    X_ = roi.utils.lstsq_reg(A, B, P=[2*I]*r)[0]
    assert la.norm(X_) <= la.norm(X)

    # Test residuals actually give the Frobenius norm squared of the misfit.
    Acond = np.linalg.cond(A)
    X_, res, rnk, svdvals = roi.utils.lstsq_reg(A, B, P=0)
    assert np.isclose(np.sum(res), np.sum((A @ X_ - B)**2))
    assert np.isclose(abs(svdvals[0]/svdvals[-1]), Acond)

    # Ensure residuals larger, condition numbers smaller with regularization.
    X_, res, rnk, svdvals = roi.utils.lstsq_reg(A, B, P=2)
    assert np.sum(res) > np.sum((A @ X_ - B)**2)
    assert abs(svdvals[0]/svdvals[-1]) < Acond

    X_, res, rnk, svdvals = roi.utils.lstsq_reg(A, B, P=[2*I]*r)
    assert np.sum(res) > np.sum((A @ X_ - B)**2)
    assert abs(svdvals[0]/svdvals[-1]) < Acond


def test_lstsq_reg(n_tests=5):
    """Test utils._solver.lstsq_reg()."""
    A = np.random.random((20,10))
    B = np.random.random((20, 5))

    # Negative regularization parameter not allowed.
    with pytest.raises(ValueError) as exc:
        roi.utils.lstsq_reg(A, B, -1)
    assert exc.value.args[0] == "regularization parameter must be nonnegative"

    # Try with underdetermined system.
    with pytest.warns(la.LinAlgWarning) as exc:
        roi.utils.lstsq_reg(A[:8,:], B[:8,:])
    assert exc[0].message.args[0] == "least squares system is underdetermined"

    # b must be one- or two-dimensional
    with pytest.raises(ValueError) as exc:
        roi.utils.lstsq_reg(A, np.random.random((2,2,2)))
    assert exc.value.args[0] == "`b` must be one- or two-dimensional"

    with pytest.raises(ValueError) as exc:
        roi.utils.lstsq_reg(A, np.array(5))
    assert exc.value.args[0] == "`b` must be one- or two-dimensional"

    with pytest.raises(ValueError) as exc:
        roi.utils.lstsq_reg(A, B[:,0],
                            [np.random.random((10,10)) for i in range(5)])
    assert exc.value.args[0] == "`b` must be two-dimensional with multiple P"

    # Badly shaped regularization matrix.
    with pytest.raises(ValueError) as exc:
        roi.utils.lstsq_reg(A, B, np.random.random((5,5)))
    assert exc.value.args[0] == \
        "P must be (d,d) with d = number of columns of A"

    # Bad number of regularization matrices (list).
    with pytest.raises(ValueError) as exc:
        roi.utils.lstsq_reg(A, B,
                            [np.random.random((10,10)) for i in range(3)])
    assert exc.value.args[0] == \
        "multiple P requires exactly r entries with r = number of columns of b"

    # Bad number of regularization matrices (generator).
    with pytest.raises(ValueError) as exc:
        roi.utils.lstsq_reg(A, B,
                            (np.random.random((10,10)) for i in range(3)))
    assert exc.value.args[0] == \
        "multiple P requires exactly r entries with r = number of columns of b"


    # Do individual tests.
    k, m = 200, 20
    for r in np.random.randint(4, 50, n_tests):
        _test_lstsq_reg_single(k, min(k, 1 + r + r*(r+1)//2 + m), r)
