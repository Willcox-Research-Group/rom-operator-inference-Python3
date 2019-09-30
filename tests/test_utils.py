# test_utils.py
"""Tests for rom_operator_inference.utils.py."""

import pytest
import numpy as np
from scipy import linalg as la

import rom_operator_inference as roi


# rom_operator_inference.utils.lstsq_reg() ------------------------------------
def _test_lstq_reg_single(k,d,r):
    """Do one test of rom_operator_inference.utils.lstsq_reg()."""
    A = np.random.random((k, d))
    b = np.random.random(k)
    B = np.random.random((k,r))
    I = np.eye(d)

    # VECTOR TEST
    x = la.lstsq(A, b)[0]

    # Ensure that the least squares solution is the usual one when G=0.
    x_ = roi.utils.lstsq_reg(A, b, G=0)[0]
    assert np.allclose(x, x_)

    # Ensure that the least squares solution is the usual one when G=0 (matrix)
    x_ = roi.utils.lstsq_reg(A, b, np.zeros((d,d)))[0]
    assert np.allclose(x, x_)

    # Check that the regularized least squares solution has decreased norm.
    x_ = roi.utils.lstsq_reg(A, b, G=2)[0]
    assert la.norm(x_) <= la.norm(x)

    x_ = roi.utils.lstsq_reg(A, b, G=2*I)[0]
    assert la.norm(x_) <= la.norm(x)

    # MATRIX TEST
    X = la.lstsq(A, B)[0]

    # Ensure that the least squares solution is the usual one when G=0.
    X_ = roi.utils.lstsq_reg(A, B, G=0)[0]
    assert np.allclose(X, X_)

    # Ensure that the least squares solution is the usual one when G=0 (matrix)
    X_ = roi.utils.lstsq_reg(A, B, G=0)[0]
    assert np.allclose(X, X_)

    # Ensure that the least squares solution is the usual one when G=0 (list)
    X_ = roi.utils.lstsq_reg(A, B, G=[np.zeros((d,d))]*r)[0]
    assert np.allclose(X, X_)

    # Check that the regularized least squares solution has decreased norm.
    X_ = roi.utils.lstsq_reg(A, B, G=2)[0]
    assert la.norm(X_) <= la.norm(X)

    X_ = roi.utils.lstsq_reg(A, B, G=2)[0]
    assert la.norm(X_) <= la.norm(X)

    X_ = roi.utils.lstsq_reg(A, B, G=[2*I]*r)[0]
    assert la.norm(X_) <= la.norm(X)


def test_lstsq_reg(n_tests=5):
    """Test rom_operator_inference.utils.lstsq_reg()."""
    A = np.random.random((20,10))
    B = np.random.random((20, 5))

    # Negative regularization parameter not allowed.
    with pytest.raises(ValueError) as exc:
        roi.utils.lstsq_reg(A, B, -1)
    assert exc.value.args[0] == "regularization parameter must be nonnegative"

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
    assert exc.value.args[0] == "`b` must be two-dimensional with multiple G"

    # Badly shaped regularization matrix.
    with pytest.raises(ValueError) as exc:
        roi.utils.lstsq_reg(A, B, np.random.random((5,5)))
    assert exc.value.args[0] == \
        "G must be (d,d) with d = number of columns of A"

    with pytest.raises(ValueError) as exc:
        roi.utils.lstsq_reg(A, B,
                            [np.random.random((10,10)) for i in range(3)])
    assert exc.value.args[0] == \
        "list G must have r entries with r = number of columns of b"

    # Do individual tests.
    k, m = 200, 20
    for r in np.random.randint(4, 50, n_tests):
        s = r*(r+1)//2
        _test_lstq_reg_single(k, r+s+m+1, r)


# rom_operator_inference.utils.kron_compact() ---------------------------------
def _test_kron_compact_single_vector(n):
    """Do one vector test of rom_operator_inference.utils.kron_compact()."""
    x = np.random.random(n)
    x2 = roi.utils.kron_compact(x)
    assert x2.ndim == 1
    assert x2.shape[0] == n*(n+1)//2
    for i in range(n):
        assert np.allclose(x2[i*(i+1)//2:(i+1)*(i+2)//2], x[i]*x[:i+1])


def _test_kron_compact_single_matrix(n):
    """Do one matrix test of rom_operator_inference.utils.kron_compact()."""
    X = np.random.random((n,n))
    X2 = roi.utils.kron_compact(X)
    assert X2.ndim == 2
    assert X2.shape[0] == n*(n+1)//2
    assert X2.shape[1] == n
    for i in range(n):
        assert np.allclose(X2[i*(i+1)//2:(i+1)*(i+2)//2], X[i]*X[:i+1])


def test_kron_compact(n_tests=100):
    """Test rom_operator_inference.utils.kron_compact()."""
    for n in np.random.randint(2, 100, n_tests):
        _test_kron_compact_single_vector(n)
        _test_kron_compact_single_matrix(n)


# rom_operator_inference.utils.F2H() ------------------------------------------
def _test_F2H_single(r):
    """Do one test of rom_operator_inference.utils.F2H()."""
    x = np.random.random(r)

    # Do a valid F2H() calculation and check dimensions.
    s = r*(r+1)//2
    F = np.random.random((r,s))
    H = roi.utils.F2H(F)
    assert H.shape == (r,r**2)

    # Check that F(x^2) == H(x⊗x).
    Hxx = H @ np.kron(x,x)
    assert np.allclose(F @ roi.utils.kron_compact(x), Hxx)

    # Check properties of the tensor for H.
    Htensor = H.reshape((r,r,r))
    assert np.allclose(Htensor @ x @ x, Hxx)
    for subH in H:
        assert np.allclose(subH, subH.T)


def test_F2H(n_tests=100):
    """Test rom_operator_inference.utils.F2H()."""
    # Try to do F2H() with a bad second dimension.
    r = 5
    sbad = r*(r+3)//2
    F = np.random.random((r, sbad))
    with pytest.raises(ValueError) as exc:
        roi.utils.F2H(F)
    assert exc.value.args[0] == \
        f"invalid shape (r,s) = {(r,sbad)} with s != r(r+1)/2"

    # Do 100 test cases of varying dimensions.
    for r in np.random.randint(2, 100, n_tests):
        _test_F2H_single(r)


# rom_operator_inference.utils.H2F() ------------------------------------------
def _test_H2F_single(r):
    """Do one test of rom_operator_inference.utils.F2H()."""
    x = np.random.random(r)

    # Do a valid H2F() calculation and check dimensions.
    H = np.random.random((r,r**2))
    s = r*(r+1)//2
    F = roi.utils.H2F(H)
    assert F.shape == (r,s)

    # Check that F(x^2) == H(x⊗x).
    Hxx = H @ np.kron(x,x)
    assert np.allclose(Hxx, F @ roi.utils.kron_compact(x))

    # Check that F2H() and H2F() are inverses up to symmetry.
    H2 = roi.utils.F2H(F)
    Ht = H.reshape((r,r,r))
    Htnew = np.empty_like(Ht)
    for l in range(r):
        Htnew[l] = (Ht[l] + Ht[l].T) / 2
    assert np.allclose(H2, Htnew.reshape(H.shape))


def test_H2F(n_tests=100):
    """Test rom_operator_inference.utils.F2H()."""
    # Try to do H2F() with a bad second dimension.
    r = 5
    r2bad = r**2 + 1
    H = np.random.random((r, r2bad))
    with pytest.raises(ValueError) as exc:
        roi.utils.H2F(H)
    assert exc.value.args[0] == \
        f"invalid shape (r,a) = {(r,r2bad)} with a != r**2"

    # Do 100 test cases of varying dimensions.
    for r in np.random.randint(2, 100, n_tests):
        _test_H2F_single(r)
