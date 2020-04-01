# test_utils.py
"""Tests for rom_operator_inference.utils.py."""

import pytest
import numpy as np
from scipy import linalg as la

import rom_operator_inference as roi


def test_get_least_squares_size():
    """Test utils.get_least_squares_size()."""
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

    # Test with inputs.
    assert roi.utils.get_least_squares_size("cB", r, m) == 1 + m
    assert roi.utils.get_least_squares_size("AB", r, m) == r + m
    assert roi.utils.get_least_squares_size("cAB", r, m) == 1 + r + m
    assert roi.utils.get_least_squares_size("AHB", r, m) == r + r*(r+1)//2 + m

    # Test with affines.
    assert roi.utils.get_least_squares_size("c", r, affines={"c":[0,0]}) == 2
    assert roi.utils.get_least_squares_size("A", r, affines={"A":[0,0]}) == 2*r


# utils.lstsq_reg() -----------------------------------------------------------
def _test_lstq_reg_single(k,d,r):
    """Do one test of utils.lstsq_reg()."""
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

    # Check that the regularized least squares solution has decreased norm.
    X_ = roi.utils.lstsq_reg(A, B, P=2)[0]
    assert la.norm(X_) <= la.norm(X)

    X_ = roi.utils.lstsq_reg(A, B, P=2)[0]
    assert la.norm(X_) <= la.norm(X)

    X_ = roi.utils.lstsq_reg(A, B, P=[2]*r)[0]
    assert la.norm(X_) <= la.norm(X)

    X_ = roi.utils.lstsq_reg(A, B, P=[2*I]*r)[0]
    assert la.norm(X_) <= la.norm(X)


def test_lstsq_reg(n_tests=5):
    """Test utils.lstsq_reg()."""
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

    with pytest.raises(ValueError) as exc:
        roi.utils.lstsq_reg(A, B,
                            [np.random.random((10,10)) for i in range(3)])
    assert exc.value.args[0] == \
        "list P must have r entries with r = number of columns of b"

    # Do individual tests.
    k, m = 200, 20
    for r in np.random.randint(4, 50, n_tests):
        _test_lstq_reg_single(k, min(k, 1 + r + r*(r+1)//2 + m), r)


# utils.kron_compact() --------------------------------------------------------
def _test_kron_compact_single_vector(n):
    """Do one vector test of utils.kron_compact()."""
    x = np.random.random(n)
    x2 = roi.utils.kron_compact(x)
    assert x2.ndim == 1
    assert x2.shape[0] == n*(n+1)//2
    for i in range(n):
        assert np.allclose(x2[i*(i+1)//2:(i+1)*(i+2)//2], x[i]*x[:i+1])


def _test_kron_compact_single_matrix(n):
    """Do one matrix test of utils.kron_compact()."""
    X = np.random.random((n,n))
    X2 = roi.utils.kron_compact(X)
    assert X2.ndim == 2
    assert X2.shape[0] == n*(n+1)//2
    assert X2.shape[1] == n
    for i in range(n):
        assert np.allclose(X2[i*(i+1)//2:(i+1)*(i+2)//2], X[i]*X[:i+1])


def test_kron_compact(n_tests=100):
    """Test utils.kron_compact()."""
    # Try with bad input.
    with pytest.raises(ValueError) as exc:
        roi.utils.kron_compact(np.random.random((3,3,3)))
    assert exc.value.args[0] == "x must be one- or two-dimensional"

    # Correct inputs.
    for n in np.random.randint(2, 100, n_tests):
        _test_kron_compact_single_vector(n)
        _test_kron_compact_single_matrix(n)


# utils.kron_col() ------------------------------------------------------------
def _test_kron_col_single_vector(n, m):
    """Do one vector test of utils.kron_col()."""
    x = np.random.random(n)
    y = np.random.random(m)
    xy = roi.utils.kron_col(x, y)
    assert xy.ndim == 1
    assert xy.shape[0] == x.shape[0] * y.shape[0]
    for i in range(n):
        assert np.allclose(xy[i*m:(i+1)*m], x[i]*y)


def _test_kron_col_single_matrix(n, k, m):
    """Do one matrix test of utils.kron_col()."""
    X = np.random.random((n,k))
    Y = np.random.random((m,k))
    XY = roi.utils.kron_col(X, Y)
    assert XY.ndim == 2
    assert XY.shape[0] == X.shape[0] * Y.shape[0]
    assert XY.shape[1] == X.shape[1]
    for i in range(n):
        assert np.allclose(XY[i*m:(i+1)*m], X[i]*Y)


def test_kron_col(n_tests=100):
    """Test utils.kron_compact()."""
    # Try with bad inputs.
    with pytest.raises(ValueError) as exc:
        roi.utils.kron_col(np.random.random(5), np.random.random((5,5)))
    assert exc.value.args[0] == \
        "x and y must have the same number of dimensions"

    x = np.random.random((3,3))
    y = np.random.random((3,4))
    with pytest.raises(ValueError) as exc:
        roi.utils.kron_col(x, y)
    assert exc.value.args[0] == "x and y must have the same number of columns"

    x, y = np.random.random((2,3,3,3))
    with pytest.raises(ValueError) as exc:
        roi.utils.kron_col(x, y)
    assert exc.value.args[0] == "x and y must be one- or two-dimensional"

    for n in np.random.randint(4, 100, n_tests):
        m = np.random.randint(2, n)
        k = np.random.randint(2, 100)
        _test_kron_col_single_vector(n, m)
        _test_kron_col_single_matrix(n, k, m)


# utils.expand_Hc() -----------------------------------------------------------
def _test_expand_Hc_single(r):
    """Do one test of utils.expand_Hc()."""
    x = np.random.random(r)

    # Do a valid expand_Hc() calculation and check dimensions.
    s = r*(r+1)//2
    Hc = np.random.random((r,s))
    H = roi.utils.expand_Hc(Hc)
    assert H.shape == (r,r**2)

    # Check that Hc(x^2) == H(x⊗x).
    Hxx = H @ np.kron(x,x)
    assert np.allclose(Hc @ roi.utils.kron_compact(x), Hxx)

    # Check properties of the tensor for H.
    Htensor = H.reshape((r,r,r))
    assert np.allclose(Htensor @ x @ x, Hxx)
    for subH in H:
        assert np.allclose(subH, subH.T)


def test_expand_Hc(n_tests=100):
    """Test utils.expand_Hc()."""
    # Try to do expand_Hc() with a bad second dimension.
    r = 5
    sbad = r*(r+3)//2
    Hc = np.random.random((r, sbad))
    with pytest.raises(ValueError) as exc:
        roi.utils.expand_Hc(Hc)
    assert exc.value.args[0] == \
        f"invalid shape (r,s) = {(r,sbad)} with s != r(r+1)/2"

    # Do 100 test cases of varying dimensions.
    for r in np.random.randint(2, 100, n_tests):
        _test_expand_Hc_single(r)


# utils.compress_H() ----------------------------------------------------------
def _test_compress_H_single(r):
    """Do one test of utils.expand_Hc()."""
    x = np.random.random(r)

    # Do a valid compress_H() calculation and check dimensions.
    H = np.random.random((r,r**2))
    s = r*(r+1)//2
    Hc = roi.utils.compress_H(H)
    assert Hc.shape == (r,s)

    # Check that Hc(x^2) == H(x⊗x).
    Hxx = H @ np.kron(x,x)
    assert np.allclose(Hxx, Hc @ roi.utils.kron_compact(x))

    # Check that expand_Hc() and compress_H() are inverses up to symmetry.
    H2 = roi.utils.expand_Hc(Hc)
    Ht = H.reshape((r,r,r))
    Htnew = np.empty_like(Ht)
    for l in range(r):
        Htnew[l] = (Ht[l] + Ht[l].T) / 2
    assert np.allclose(H2, Htnew.reshape(H.shape))


def test_compress_H(n_tests=100):
    """Test utils.expand_Hc()."""
    # Try to do compress_H() with a bad second dimension.
    r = 5
    r2bad = r**2 + 1
    H = np.random.random((r, r2bad))
    with pytest.raises(ValueError) as exc:
        roi.utils.compress_H(H)
    assert exc.value.args[0] == \
        f"invalid shape (r,a) = {(r,r2bad)} with a != r**2"

    # Do 100 test cases of varying dimensions.
    for r in np.random.randint(2, 100, n_tests):
        _test_compress_H_single(r)
